#!/usr/bin/env python3
"""Atelier VidGen — R2 exchange protocol (shared by bridge, mock worker, Kaggle kernel).

Layout inside bucket under R2_PREFIX (default "vidgen"):
  jobs/<id>/job.json      pending job spec (existence = queued)
  jobs/<id>/ref.jpg|png   i2v reference frame
  status/<id>.json        heartbeat / result status (RENDERING|DONE|FAILED)
  results/<id>/video.mp4  finished clip
  results/<id>/thumb.jpg  poster frame

Claim protocol: delete jobs/<id>/job.json — in S3 semantics exactly one
deleter wins, so N workers never double-render.
"""
import json
import os
import posixpath
import time
from typing import Any, Dict, List, Optional

ENV_PATH = os.environ.get("VIDGEN_ENV", "/home/z/my-project/scripts/vidgen_runtime.env")


def load_env() -> Dict[str, str]:
    env = dict(os.environ)
    if os.path.exists(ENV_PATH):
        for line in open(ENV_PATH):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k] = v
    return env


class Exchange:
    def __init__(self) -> None:
        import boto3  # lazy import so the panel host doesn't need it
        from botocore.config import Config

        env = load_env()
        self.prefix = env.get("R2_PREFIX", "vidgen").strip("/")
        self.bucket = env["R2_BUCKET"]
        self.s3 = boto3.client(
            "s3",
            endpoint_url=env["R2_ENDPOINT"],
            aws_access_key_id=env["R2_KEY"],
            aws_secret_access_key=env["R2_SECRET"],
            config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            region_name="auto",
        )

    # ── key helpers ──────────────────────────────────────────────
    def _k(self, *parts: str) -> str:
        return posixpath.join(self.prefix, *parts)

    # ── basic ops ────────────────────────────────────────────────
    def put_json(self, key: str, obj: Dict[str, Any]) -> None:
        self.s3.put_object(
            Bucket=self.bucket, Key=key,
            Body=json.dumps(obj).encode(), ContentType="application/json",
        )

    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        try:
            return json.loads(self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read())
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception:
            return None

    def put_file(self, key: str, path: str, content_type: str = "application/octet-stream") -> None:
        ct = content_type
        if ct == "application/octet-stream":
            if path.endswith(".mp4"):
                ct = "video/mp4"
            elif path.endswith(".jpg") or path.endswith(".jpeg"):
                ct = "image/jpeg"
            elif path.endswith(".png"):
                ct = "image/png"
        self.s3.upload_file(Bucket=self.bucket, Key=key, Filename=path, ExtraArgs={"ContentType": ct})

    def get_file(self, key: str, path: str) -> bool:
        try:
            self.s3.download_file(Bucket=self.bucket, Key=key, Filename=path)
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            return True
        except Exception:
            return False

    def list(self, sub: str) -> List[str]:
        out: List[str] = []
        paginator = self.s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self._k(sub)):
            for obj in page.get("Contents", []):
                out.append(obj["Key"])
        return out

    # ── protocol ops ─────────────────────────────────────────────
    def submit_job(self, spec: Dict[str, Any], ref_path: Optional[str] = None) -> None:
        jid = spec["id"]
        self.put_json(self._k("jobs", jid, "job.json"), spec)
        if ref_path:
            ext = "png" if ref_path.lower().endswith(".png") else "jpg"
            self.put_file(self._k("jobs", jid, f"ref.{ext}"), ref_path)

    def pending_jobs(self) -> List[Dict[str, Any]]:
        """List queued job specs (job.json present = not yet claimed)."""
        jobs: List[Dict[str, Any]] = []
        for key in self.list("jobs"):
            if key.endswith("/job.json"):
                body = self.get_json(key)
                if body:
                    body["_key"] = key
                    jobs.append(body)
        jobs.sort(key=lambda j: j.get("created_at", ""))
        return jobs

    def claim(self, jid: str, key: str) -> Optional[Dict[str, Any]]:
        """Delete-as-claim: winner keeps the spec, losers get None."""
        body = self.get_json(key)
        if body is None:
            return None
        self.delete(key)
        return body

    def heartbeat(self, jid: str, status: str, *, gpu: str = "", progress: int = 0,
                  elapsed_s: float = 0, error: str = "") -> None:
        self.put_json(self._k("status", f"{jid}.json"), {
            "id": jid, "status": status, "gpu": gpu, "progress": progress,
            "elapsed_s": round(elapsed_s, 1), "error": error, "updated_at": time.time(),
        })

    def status(self, jid: str) -> Optional[Dict[str, Any]]:
        return self.get_json(self._k("status", f"{jid}.json"))

    def publish_result(self, jid: str, video_path: str, thumb_path: str,
                       gpu: str, elapsed_s: float) -> None:
        self.put_file(self._k("results", jid, "video.mp4"), video_path)
        self.put_file(self._k("results", jid, "thumb.jpg"), thumb_path)
        self.heartbeat(jid, "DONE", gpu=gpu, progress=100, elapsed_s=elapsed_s)

    def fail(self, jid: str, error: str, gpu: str = "") -> None:
        self.heartbeat(jid, "FAILED", gpu=gpu, error=error[:500])


if __name__ == "__main__":
    ex = Exchange()
    print(f"exchange ok: bucket={ex.bucket} prefix={ex.prefix}")
    print("pending jobs:", len(ex.pending_jobs()))
