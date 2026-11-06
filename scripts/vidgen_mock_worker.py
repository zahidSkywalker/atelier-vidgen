#!/usr/bin/env python3
"""Atelier VidGen mock worker — R2-native worker with the EXACT protocol of the
Kaggle kernel, minus the GPU: claims a job from the exchange, renders a 5 s
synthetic clip with ffmpeg (same dims/fps/duration/audio layout), publishes
result + heartbeat. Used to verify the full panel↔exchange↔worker loop on CPU.

Run: python3 scripts/vidgen_mock_worker.py [--once]
"""
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vidgen_exchange import Exchange  # noqa: E402

GPU_NAME = "mock:cpu-ffmpeg"
ASPECTS = {"portrait": (480, 864), "landscape": (864, 480)}


def log(msg: str) -> None:
    print(f"[mock-worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def render(spec: dict, out_mp4: str, out_thumb: str) -> None:
    w, h = ASPECTS.get(spec.get("aspect", "portrait"), ASPECTS["portrait"])
    dur = 5
    audio = bool(spec.get("audio", True))
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=24:duration={dur}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=220:duration={dur}"]
    cmd += ["-t", str(dur), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "24"]
    if audio:
        cmd += ["-c:a", "aac", "-b:a", "96k", "-shortest"]
    cmd += ["-movflags", "+faststart", out_mp4]
    subprocess.run(cmd, check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", out_mp4,
                    "-vf", "select=eq(n\\,24)", "-frames:v", "1", out_thumb], check=True)


def main() -> None:
    once = "--once" in sys.argv
    ex = Exchange()
    log(f"mock worker up — polling exchange (bucket={ex.bucket}, prefix={ex.prefix})")
    while True:
        jobs = ex.pending_jobs()
        if not jobs:
            if once:
                log("queue empty, exiting (--once)")
                return
            time.sleep(3)
            continue
        spec = jobs[0]
        jid = spec["id"]
        t0 = time.time()
        try:
            claimed = ex.claim(jid, spec["_key"])
            if claimed is None:
                log(f"{jid[:10]} claimed by another worker, skip")
                continue
            log(f"claimed {jid[:10]} ({spec['mode']}, {spec['aspect']})")
            ex.heartbeat(jid, "RENDERING", gpu=GPU_NAME, progress=10)
            with tempfile.TemporaryDirectory() as td:
                mp4 = os.path.join(td, "video.mp4")
                thumb = os.path.join(td, "thumb.jpg")
                render(spec, mp4, thumb)
                ex.heartbeat(jid, "RENDERING", gpu=GPU_NAME, progress=70)
                ex.publish_result(jid, mp4, thumb, GPU_NAME, time.time() - t0)
            log(f"{jid[:10]} DONE in {time.time() - t0:.1f}s")
        except Exception as e:
            log(f"{jid[:10]} FAILED: {e}")
            try:
                ex.fail(jid, str(e), gpu=GPU_NAME)
            except Exception:
                pass
        if once:
            return


if __name__ == "__main__":
    main()
