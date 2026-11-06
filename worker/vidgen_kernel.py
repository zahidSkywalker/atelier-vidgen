#!/usr/bin/env python3
"""
Atelier VidGen — Kaggle GPU worker kernel (Kandinsky 6.0 Video Lite).

Runs on Kaggle (account B, xion678) T4 session. Polls the R2 exchange for
queued jobs, claims them (delete-as-claim), renders 5 s clips with the
Kandinsky-6.0-Lite-distill Diffusers pipeline, publishes mp4 + thumb back.

Dual-T4 policy v5: ONE renderer process. v4 ran one worker per card; each
holds a ~15 GB resident pipeline in system RAM, the T4x2 box has 29 GB, and
the second load triggered the kernel OOM-killer which SIGKILL'd the sibling
mid-load (frozen at its pre-load pinger mark, no traceback). RAM — not VRAM —
is the ceiling. Jobs render strictly serially on card 0; card 1 stays on
standby for a future streaming/smaller-weights path.

Quota discipline:
  - exits when the queue stays empty for IDLE_EXIT_MIN minutes
  - hard stop at MAX_SESSION_MIN minutes of wall time
Requires dataset: xion678/vidgen-config (vidgen_runtime.env with R2 creds).

T4 (sm75) notes:
  - No bf16 GEMM on Turing → load bf16 (exact), then cast transformer/text
    encoders to fp16 and VAE / audio VAE / vocoder to fp32 (bf16→fp32 is
    lossless; fp16 overflow guards live in the output sanity check).
  - No FlashAttention/SageAttention → SDPA math backend, slower but correct.
  - Output guard rejects NaN / all-black frames instead of publishing them.

v2.5 changelog:
  - single renderer process (RAM ceiling — see policy note above)
  - sequential cpu offload first in the ladder; model cpu offload demoted to
    fallback (proven silently ignored by Kandinsky6TI2VAPipeline in v1 and
    v4: its internals move the whole suite to VRAM → 14.5 GiB resident →
    i2v CUDA OOM)
"""
import os
import subprocess
import sys
import time

# ── runtime deps (installed once in main(), not re-run by spawned children) ──
def sh(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True)


def install_deps() -> None:
    print("[vidgen] installing deps…", flush=True)
    sh("pip install -q boto3 diffusers transformers accelerate safetensors "
       "soundfile imageio imageio-ffmpeg opencv-python-headless ftfy sentencepiece")


CONFIG_DIR_CANDIDATES = [
    "/kaggle/input/datasets/xion678/vidgen-config",  # new Kaggle mount layout
    "/kaggle/input/vidgen-config",                   # legacy layout fallback
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
# T4 16 GB: fight fragmentation before torch initializes CUDA in any process
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

IDLE_EXIT_MIN = int(ENV.get("IDLE_EXIT_MIN", "10"))
MAX_SESSION_MIN = int(ENV.get("MAX_SESSION_MIN", "300"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CONFIG_DIR_CANDIDATES[0])  # dataset also carries vidgen_exchange.py
try:
    from vidgen_exchange import Exchange
except ImportError:
    raise SystemExit("vidgen_exchange.py not found — bundle it into xion678/vidgen-config")

T0 = time.time()


class Pinger:
    """Heartbeats RENDERING every 60 s so the bridge reaper never requeues a
    live render (the 35 GB model download can outlast the 15-min stale mark)."""

    def __init__(self, ex: Exchange, jid: str, gpu: str) -> None:
        self.ex, self.jid, self.gpu = ex, jid, gpu
        self.state = {"progress": 20}
        self._stop = False
        import threading
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def set(self, progress: int) -> None:
        self.state["progress"] = max(self.state["progress"], progress)

    def _run(self) -> None:
        while not self._stop:
            try:
                self.ex.heartbeat(self.jid, "RENDERING", gpu=self.gpu,
                                  progress=self.state["progress"],
                                  elapsed_s=time.time() - T0)
            except Exception:
                pass
            time.sleep(60)

    def stop(self) -> None:
        self._stop = True


def _has_fp16(repo: str) -> bool:
    try:
        from huggingface_hub import model_info
        files = [f.rfilename for f in model_info(repo, files_metadata=False).siblings or []]
        return any("fp16" in f for f in files)
    except Exception:
        return False


def load_pipeline(gpu: str):
    import torch
    from diffusers import Kandinsky6TI2VAPipeline

    # SERIALIZE the 35 GB download/load across cards: two concurrent
    # from_pretrained calls peaked ~2x Kaggle RAM and the OOM killer
    # silently reaped the second process (v1 post-mortem). The lock makes
    # card 1 wait, then load from the warm HF cache in ~2 min.
    import fcntl
    lock_path = "/tmp/vidgen_model_load.lock"
    lock_f = open(lock_path, "w")
    print(f"[{gpu}] waiting for model-load lock…", flush=True)
    fcntl.flock(lock_f, fcntl.LOCK_EX)
    try:
        repo = ENV.get("KANDINSKY_REPO", "kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers")
        print(f"[{gpu}] downloading/loading {repo} (bf16, exact)…", flush=True)
        pipe = Kandinsky6TI2VAPipeline.from_pretrained(
            repo, torch_dtype=torch.bfloat16, variant="fp16" if _has_fp16(repo) else None,
            low_cpu_mem_usage=True,  # stream shards; no full-state-dict RAM spike
        )
        # Turing casting: DiT + text encoders → fp16 (compute), decoders → fp32 (safe)
        for comp in ("transformer", "text_encoder", "text_encoder_2"):
            m = getattr(pipe, comp, None)
            if m is not None:
                m.to(torch.float16)
        for comp in ("vae", "audio_vae", "vocoder"):
            m = getattr(pipe, comp, None)
            if m is not None:
                m.to(torch.float32)
        # memory ladder v5: sequential offload FIRST — enable_model_cpu_offload
        # is silently ignored by Kandinsky6TI2VAPipeline (v1 AND v4 post-mortems:
        # its internals move the whole suite to VRAM, 14.5 GiB resident, i2v
        # OOM). Accelerate's per-layer hooks always work. all-GPU is the very
        # last resort and effectively never viable on a 16 GB card.
        strategy = "all-gpu"
        for fn, limit in (("enable_sequential_cpu_offload", 1.0), ("enable_model_cpu_offload", 4.0)):
            try:
                getattr(pipe, fn)()
                torch.cuda.synchronize()
                resident = torch.cuda.memory_allocated() / 2**30
                print(f"[{gpu}] {fn}: resident after apply = {resident:.2f} GiB", flush=True)
                if resident < limit:
                    strategy = fn
                    break
                print(f"[{gpu}] {fn} ignored by pipeline (still resident) — escalating", flush=True)
            except Exception as e:
                print(f"[{gpu}] {fn} unavailable: {type(e).__name__}: {e}", flush=True)
        if strategy == "all-gpu":
            pipe.to("cuda")
        print(f"[{gpu}] memory strategy: {strategy}", flush=True)
        try:
            pipe.vae.enable_tiling()  # decode headroom on 16 GB
        except Exception:
            pass
        return pipe
    finally:
        fcntl.flock(lock_f, fcntl.LOCK_UN)  # release on success AND failure
        print(f"[{gpu}] model-load lock released", flush=True)


def _audio_to_np(audio, ):
    import numpy as np
    import torch
    if audio is None:
        return None
    if isinstance(audio, (list, tuple)):
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().float().cpu().numpy()
    arr = np.asarray(audio, dtype=np.float32).squeeze()
    return arr if arr.size else None


def worker_loop(gpu_index: int) -> None:
    gpu = f"t4:{gpu_index}"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    import torch
    ex = Exchange()
    pipe = None
    audio_sr = None

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
        pinger = None
        try:
            claimed = ex.claim(jid, spec["_key"])
            if claimed is None:
                continue
            print(f"[{gpu}] claimed {jid[:10]} ({spec['mode']})", flush=True)
            pinger = Pinger(ex, jid, gpu)
            pinger.set(15)

            if pipe is None:
                pinger.set(30)
                pipe = load_pipeline(gpu)
                audio_sr = getattr(pipe, "audio_sample_rate", 44100)
            pinger.set(40)

            dims = {"portrait": (480, 864), "landscape": (864, 480)}
            h, w = dims.get(spec.get("aspect", "portrait"), dims["portrait"])
            prompt = spec["prompt"]
            kwargs = dict(
                prompt=prompt,
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

            steps = kwargs["num_inference_steps"]

            def _step_cb(p, i, t, cbk):
                pinger.set(40 + int(35 * (i + 1) / steps))
                return {}

            # SINGLE introspected call — v2.1 lesson: each rejected pipe()
            # attempt leaves its offload-moved modules resident, so retries
            # pack the card and the real render OOMs. Filter kwargs by
            # inspecting the signature BEFORE the first and only call.
            import inspect
            sig = inspect.signature(pipe.__call__).parameters
            call = dict(kwargs)
            if "callback_on_step_end" in sig:
                call["callback_on_step_end"] = _step_cb
            if "expand_prompts" in sig and len(prompt) < 200:
                call["expand_prompts"] = True
            i2v = spec["mode"] == "i2v" and spec.get("ref_key_ext")
            if i2v and "image" in sig:
                from PIL import Image
                img = Image.open(ref_local_path(jid, spec["ref_key_ext"])).convert("RGB")
                call["image"] = img.resize((w, h), Image.LANCZOS)  # pre-fit: lighter VAE encode
            elif i2v:
                print(f"[{gpu}] pipeline has no image= param — falling back to T2V", flush=True)
            missing = [k for k in ("height", "width", "num_frames", "num_inference_steps", "guidance_scale") if k not in sig]
            if missing:
                raise RuntimeError(f"pipeline signature lacks core kwargs: {missing}")
            supported = {k: v for k, v in call.items() if k in sig}
            out = pipe(**supported)

            frames = out.frames[0]
            audio_np = _audio_to_np(getattr(out, "audio", None))
            pinger.set(80)

            # honest-fail guard: never publish NaN / black frames (imagegen lesson)
            import numpy as np
            arr = np.asarray(frames, dtype=np.float32)
            if arr.size == 0 or not np.isfinite(arr).all():
                raise RuntimeError("degenerate output: non-finite frames (fp16 overflow suspected)")
            if float(arr.std()) < 1e-3:
                raise RuntimeError("degenerate output: flat/black frames")

            import imageio
            workdir = f"/tmp/vidgen_{jid[:10]}"
            os.makedirs(workdir, exist_ok=True)
            silent = os.path.join(workdir, "silent.mp4")
            imageio.mimwrite(silent, np.stack(frames), fps=24, codec="libx264", quality=8,
                             macro_block_size=16)

            final = silent
            if spec.get("audio", True) and audio_np is not None:
                import soundfile as sf
                wav = os.path.join(workdir, "audio.wav")
                sf.write(wav, audio_np, audio_sr or 44100)
                final = os.path.join(workdir, "video.mp4")
                sh(f'ffmpeg -y -loglevel error -i "{silent}" -i "{wav}" -c:v copy -c:a aac -b:a 160k -shortest -movflags +faststart "{final}"')
            else:
                final = os.path.join(workdir, "video_mute.mp4")
                sh(f'ffmpeg -y -loglevel error -i "{silent}" -an -c:v copy -movflags +faststart "{final}"')

            thumb = os.path.join(workdir, "thumb.jpg")
            sh(f'ffmpeg -y -loglevel error -i "{final}" -vf "select=eq(n\\,24)" -frames:v 1 "{thumb}"')
            ex.publish_result(jid, final, thumb, gpu, time.time() - t0)
            pinger.stop()
            pinger = None
            print(f"[{gpu}] {jid[:10]} DONE in {time.time() - t0:.0f}s", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                ex.fail(jid, str(e), gpu=gpu)
            except Exception:
                pass
            print(f"[{gpu}] {jid[:10]} FAILED: {e}", flush=True)
        finally:
            if pinger is not None:
                pinger.stop()
            try:
                import torch as _t
                _t.cuda.empty_cache()  # release scratch between jobs
            except Exception:
                pass


def ref_local_path(jid: str, ext: str) -> str:
    # ref frames are downloaded with the job by claim(); fallback fetch:
    p = f"/tmp/vidgen_{jid[:10]}/ref.{ext}"
    if not os.path.exists(p):
        ex = Exchange()
        ex.get_file(f"{ex.prefix}/jobs/{jid}/ref.{ext}", p)
    return p


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


def main() -> None:
    install_deps()
    import torch
    n_gpus = torch.cuda.device_count()
    print(f"[vidgen] {n_gpus} CUDA device(s) visible", flush=True)
    if n_gpus == 0:
        raise SystemExit("No CUDA device — attach a GPU accelerator to this kernel.")
    # v5 post-mortem: dual in-process workers each hold a ~15 GB resident
    # pipeline; the T4x2 box has 29 GB RAM, so when card 1 loaded while card 0
    # still held its pipeline, the kernel OOM-killer SIGKILL'd card 0 mid-load
    # (frozen at the pre-load pinger mark, no traceback). RAM, not VRAM, is
    # the ceiling — so ONE renderer process, jobs strictly serial, card 1 on
    # standby for a future smaller-weights path.
    print(f"[vidgen] v5: single renderer mode (RAM ceiling) — card 1+ standby", flush=True)
    worker_loop(0)


if __name__ == "__main__":
    main()
