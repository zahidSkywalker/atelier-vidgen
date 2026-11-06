#!/usr/bin/env python3
"""
Atelier VidGen — Kaggle GPU worker kernel (Kandinsky 6.0 Video Lite).

Runs on Kaggle (account B, xion678) T4 session. Polls the R2 exchange for
queued jobs, claims them (delete-as-claim), renders 5 s clips with the
Kandinsky-6.0-Lite-distill Diffusers pipeline, publishes mp4 + thumb back.

Dual-T4 policy (sir's directive): with 2 visible GPUs we spawn TWO worker
processes, one per card (CUDA_VISIBLE_DEVICES pinned) — 2x queue throughput.
Model-parallel split was rejected: a 3B model fits one 16 GB card, and
cross-card pipeline sync would add overhead without a throughput win.

Quota discipline:
  - exits when the queue stays empty for IDLE_EXIT_MIN minutes
  - hard stop at MAX_SESSION_MIN minutes of wall time
Requires dataset: xion678/vidgen-config (vidgen_runtime.env with R2 creds).
"""
import os
import subprocess
import sys
import time

# ── runtime deps (kernel starts lean) ───────────────────────────
def sh(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)

print("[vidgen] installing deps…", flush=True)
sh("pip install -q boto3 diffusers transformers accelerate safetensors soundfile imageio imageio-ffmpeg opencv-python-headless")

import json
import threading
import multiprocessing as mp

CONFIG_DIR_CANDIDATES = [
    "/kaggle/input/vidgen-config",
    os.path.expanduser("~/.vidgen"),
    ".",
]

def find_config_file() -> str:
    for d in CONFIG_DIR_CANDIDATES:
        p = os.path.join(d, "vidgen_runtime.env")
        if os.path.exists(p):
            return p
    raise SystemExit("vidgen_runtime.env not found (dataset xion678/vidgen-config attached?)")

CFG_PATH = find_config_file()
ENV = dict(os.environ)
for line in open(CFG_PATH):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        ENV[k] = v
os.environ.update({k: ENV[k] for k in ("R2_ACCOUNT", "R2_KEY", "R2_SECRET", "R2_BUCKET", "R2_PREFIX", "R2_ENDPOINT") if k in ENV})

IDLE_EXIT_MIN = int(ENV.get("IDLE_EXIT_MIN", "10"))
MAX_SESSION_MIN = int(ENV.get("MAX_SESSION_MIN", "300"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CONFIG_DIR_CANDIDATES[0])  # dataset also carries vidgen_exchange.py
try:
    from vidgen_exchange import Exchange
except ImportError:
    raise SystemExit("vidgen_exchange.py not found — bundle it into xion678/vidgen-config")

T0 = time.time()


def worker_loop(gpu_index: int) -> None:
    gpu = f"t4:{gpu_index}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import torch
    ex = Exchange()

    pipe = None
    audio_sr = None

    def load_pipeline():
        from diffusers import Kandinsky6TI2VAPipeline
        import torch
        nonlocal pipe, audio_sr
        if pipe is not None:
            return
        print(f"[{gpu}] loading Kandinsky-6.0-Lite-distill-5s-Diffusers (first job)…", flush=True)
        repo = ENV.get("KANDINSKY_REPO", "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers")
        pipe = Kandinsky6TI2VAPipeline.from_pretrained(
            repo, torch_dtype=torch.float16, variant="fp16" if _has_fp16(repo) else None,
        )
        # T4 16 GB: module offload keeps the DiT resident, encoders/VAE stream
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
        audio_sr = getattr(pipe, "audio_sample_rate", 44100)
        print(f"[{gpu}] pipeline ready", flush=True)

    def _has_fp16(repo: str) -> bool:
        try:
            from huggingface_hub import model_info
            files = [f.rfilename for f in model_info(repo, files_metadata=False).siblings or []]
            return any("fp16" in f for f in files)
        except Exception:
            return False

    idle_since = None
    while True:
        if (time.time() - T0) > MAX_SESSION_MIN * 60:
            print(f"[{gpu}] session cap {MAX_SESSION_MIN} min reached — exiting", flush=True)
            return
        jobs = ex.pending_jobs()
        if not jobs:
            if idle_since is None:
                idle_since = time.time()
            elif (time.time() - idle_since) > IDLE_EXIT_MIN * 60:
                print(f"[{gpu}] idle {IDLE_EXIT_MIN} min — exiting (quota discipline)", flush=True)
                return
            time.sleep(20)
            continue
        idle_since = None
        spec = jobs[gpu_index % len(jobs)]  # with 2 GPUs each takes a different head job
        jid = spec["id"]
        t0 = time.time()
        try:
            claimed = ex.claim(jid, spec["_key"])
            if claimed is None:
                continue
            print(f"[{gpu}] claimed {jid[:10]} ({spec['mode']})", flush=True)
            ex.heartbeat(jid, "RENDERING", gpu=gpu, progress=15)

            load_pipeline()
            import torch

            dims = {"portrait": (480, 864), "landscape": (864, 480)}
            h, w = dims.get(spec.get("aspect", "portrait"), dims["portrait"])
            kwargs = dict(
                prompt=spec["prompt"],
                height=h,
                width=w,
                num_frames=121,
                num_inference_steps=10,
                guidance_scale=1.0,
            )
            generator = None
            if spec.get("seed") is not None:
                generator = torch.Generator("cuda").manual_seed(int(spec["seed"]))
                kwargs["generator"] = generator

            ex.heartbeat(jid, "RENDERING", gpu=gpu, progress=40)
            out = None
            if spec["mode"] == "i2v" and spec.get("ref_key_ext"):
                import imageio.v2 as iio
                ref = iio.imread(ref_local_path(jid, spec["ref_key_ext"]))
                try:
                    out = pipe(image=ref, **kwargs)
                except TypeError:
                    print(f"[{gpu}] pipeline refused image= kwarg — falling back to T2V", flush=True)
                    out = pipe(**kwargs)
            else:
                out = pipe(**kwargs)

            frames = out.frames[0]
            audio = getattr(out, "audio", None)
            ex.heartbeat(jid, "RENDERING", gpu=gpu, progress=80)

            import imageio
            import numpy as np
            workdir = f"/tmp/vidgen_{jid[:10]}"
            os.makedirs(workdir, exist_ok=True)
            silent = os.path.join(workdir, "silent.mp4")
            imageio.mimwrite(silent, np.stack(frames), fps=24, codec="libx264", quality=8,
                             macro_block_size=16)

            final = silent
            if spec.get("audio", True) and audio is not None and len(audio):
                import soundfile as sf
                wav = os.path.join(workdir, "audio.wav")
                sf.write(wav, np.asarray(audio), audio_sr or 44100)
                final = os.path.join(workdir, "video.mp4")
                sh(f'ffmpeg -y -loglevel error -i "{silent}" -i "{wav}" -c:v copy -c:a aac -b:a 160k -shortest -movflags +faststart "{final}"')
            else:
                final = os.path.join(workdir, "video_mute.mp4")
                sh(f'ffmpeg -y -loglevel error -i "{silent}" -an -c:v copy -movflags +faststart "{final}"')

            thumb = os.path.join(workdir, "thumb.jpg")
            sh(f'ffmpeg -y -loglevel error -i "{final}" -vf "select=eq(n\\,24)" -frames:v 1 "{thumb}"')
            ex.publish_result(jid, final, thumb, gpu, time.time() - t0)
            print(f"[{gpu}] {jid[:10]} DONE in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                ex.fail(jid, str(e), gpu=gpu)
            except Exception:
                pass
            print(f"[{gpu}] {jid[:10]} FAILED: {e}", flush=True)


def ref_local_path(jid: str, ext: str) -> str:
    # ref frames are downloaded with the job by claim(); fallback fetch:
    p = f"/tmp/vidgen_{jid[:10]}/ref.{ext}"
    if not os.path.exists(p):
        ex = Exchange()
        ex.get_file(f"{ex.prefix}/jobs/{jid}/ref.{ext}", p)
    return p


def claim_with_ref(ex: Exchange, spec: dict, gpu_index: int) -> None:
    """Download the i2v reference frame right after claiming."""
    jid = spec["id"]
    for ext in ("jpg", "png"):
        p = f"/tmp/vidgen_{jid[:10]}/ref.{ext}"
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if ex.get_file(f"{ex.prefix}/jobs/{jid}/ref.{ext}", p):
            spec["ref_key_ext"] = ext
            return


# patch claim to also fetch ref for i2v
_orig_claim = Exchange.claim
def _claim_with_ref(self, jid: str, key: str):
    body = _orig_claim(self, jid, key)
    if body and body.get("mode") == "i2v":
        for ext in ("jpg", "png"):
            p = f"/tmp/vidgen_{jid[:10]}/ref.{ext}"
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if self.get_file(f"{self.prefix}/jobs/{jid}/ref.{ext}", p):
                body["ref_key_ext"] = ext
                break
    return body
Exchange.claim = _claim_with_ref


if __name__ == "__main__":
    try:
        import torch
        n_gpus = torch.cuda.device_count()
    except Exception:
        n_gpus = 0
    print(f"[vidgen] {n_gpus} CUDA device(s) visible", flush=True)
    if n_gpus == 0:
        raise SystemExit("No CUDA device — attach a GPU accelerator to this kernel.")
    if n_gpus == 1:
        worker_loop(0)
    else:
        procs = []
        for i in range(min(n_gpus, 2)):  # dual-T4: one worker per card
            p = mp.Process(target=worker_loop, args=(i,))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
