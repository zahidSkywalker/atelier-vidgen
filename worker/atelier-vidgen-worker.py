#!/usr/bin/env python3
"""
Atelier VidGen — Kaggle GPU worker kernel (LTX-2.5 22B distilled, GGUF) — v7.

v7 = dual-lane orchestrator + 15s combined mode (TASK-20261107-04, sir approved).
Engine : LTX-2.5 22B audio-video DiT, distilled variant, Unsloth UD-Q4_K_XL GGUF
         (13.0 GB, VRAM-resident on a T4 — no offload crawl).
Stack  : ComfyUI @ 2f35f4a08 + city96/ComfyUI-GGUF @ 6ea2651e
         + pollockjj/ComfyUI-MultiGPU (DisTorch2, pinned rev), headless,
         driven over the local HTTP API with a programmatic workflow graph
         cloned from the official Comfy-Org LTX-2.5 templates.
Lanes  : lane0 = ComfyUI on card 0 (sees both cards — hosts the 15s split
         graph), lane1 = ComfyUI on card 1 (5s overflow lane). Regular 5s
         jobs run two parallel lanes; a 15s job runs alone across both
         cards (DiT cuda:0, TE+VAE cuda:1 via DisTorch2, overflow streams
         over PCIe — never CPU/RAM offload). Single-GPU boxes degrade to
         lane0 only.
Files  : all weights fetched from unsloth/LTX-2.5-GGUF (ungated mirror):
           distilled/ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf
           text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
           vae/ltx-2.5-video-vae-conv-bf16.safetensors   (low-memory conv VAE)
           vae/ltx-2.5-audio-vae-bf16.safetensors
           latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0
Policy : gemma-12B TE is recast to ComfyUI mixed-ops fp8q on boot (v18
         lesson: bf16 TE sits exactly on the 29 GB RAM cgroup edge; fp8q
         halves resident weights, dequant per-op in fp16 is Turing-safe).
         Sampling is two-stage (8 base steps + 3 refine steps at 2x latent
         upscale, euler_ancestral, dual-CFG 1/1, distilled schedules baked
         into ManualSigmas). On CUDA OOM a 15s job retries once at a
         dropped resolution bucket (864x480 -> 720x400 class).
Quota  : exits when the queue stays empty for IDLE_EXIT_MIN minutes; hard
         stop at MAX_SESSION_MIN minutes. Heartbeats every 60 s.
Creds  : dataset jahidul007/vidgen-config → vidgen_runtime.env (+ vidgen_exchange.py).
"""
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error

# ────────────────────────── config & exchange ──────────────────────────

CONFIG_DIR_CANDIDATES = [
    "/kaggle/input/datasets/jahidul007/vidgen-config",
    "/kaggle/input/vidgen-config",
    os.path.expanduser("~/.vidgen"),
    ".",
]


def find_config_file() -> str:
    for d in CONFIG_DIR_CANDIDATES:
        p = os.path.join(d, "vidgen_runtime.env")
        if os.path.exists(p):
            return p
    raise SystemExit("vidgen_runtime.env not found (dataset jahidul007/vidgen-config attached?)")


CFG_PATH = find_config_file()
ENV = dict(os.environ)
for line in open(CFG_PATH):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        ENV[k] = v
for k in ("R2_ACCOUNT", "R2_KEY", "R2_SECRET", "R2_BUCKET", "R2_PREFIX", "R2_ENDPOINT", "HF_TOKEN"):
    if k in ENV:
        os.environ[k] = ENV[k]

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

IDLE_EXIT_MIN = int(ENV.get("IDLE_EXIT_MIN", "10"))
MAX_SESSION_MIN = int(ENV.get("MAX_SESSION_MIN", "300"))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CONFIG_DIR_CANDIDATES[0])
try:
    from vidgen_exchange import Exchange
except ImportError:
    raise SystemExit("vidgen_exchange.py not found — bundle it into jahidul007/vidgen-config")

T0 = time.time()

BASE = "/kaggle/temp/comfy"
COMFY_DIR = f"{BASE}/ComfyUI"
COMFY_REV = "2f35f4a08176d993cded35dac3332be4f7287f41"
GGUF_PLUGIN_REV = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
MGPU_PLUGIN_REV = "37a73feda5804045e4f39d8c640905d62a2169d1"

HF_REPO = "unsloth/LTX-2.5-GGUF"
GGUF_NAME = "ltx-2.5-22b-distilled-UD-Q4_K_XL.gguf"
GGUF_SUB = "distilled"
TE_NAME = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
TE_SUB = "text_encoders"
VVAE_NAME = "ltx-2.5-video-vae-conv-bf16.safetensors"
AVAE_NAME = "ltx-2.5-audio-vae-bf16.safetensors"
VAE_SUB = "vae"
UP_NAME = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
UP_SUB = "latent_upscale_models"

NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly"
SIGMAS_BASE = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"  # 8 steps
SIGMAS_REFINE = "0.85, 0.7250, 0.4219, 0.0"                                            # 3 steps
FPS = 24
DURATIONS = (5, 15)
ASPECTS = {"landscape": (864, 480), "portrait": (480, 864)}
OOM_ASPECTS = {"landscape": (720, 400), "portrait": (400, 720)}  # dropped bucket (v7 fallback)

TE_LOADED_NAME = TE_NAME  # switched to the fp8q variant after boot conversion


# Regex over the transformer-body Linear weights (the RAM bulk). Everything
# else — embeddings, norms, biases, vision/audio towers, hf_asset blobs —
# stays fp16/passthrough on purpose: MixedPrecisionOps only quantizes Linear.
QUANT_LIN_RE = re.compile(
    r"^model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj)"
    r"|per_layer_input_gate|per_layer_projection)\.weight$"
    r"|^model\.per_layer_model_projection\.weight$")


def convert_te_fp8() -> str:
    """bf16 TE is 24.5 GiB; the fp16 recast (v18) still died at the encode
    forward — 24.5 GiB + torch runtime sits exactly on the box's cgroup edge
    and swapon is not permitted on Kaggle. Fix: convert the transformer-body
    Linears to ComfyUI's MixedPrecisionOps format — fp8_e4m3fn storage +
    per-tensor weight_scale + per-layer comfy_quant metadata → TE resident
    drops to ~half, weights dequant per-op in fp16 (Turing-safe). Streaming
    two-pass writer in a throwaway subprocess: peak RAM < 3 GiB and the
    glibc arenas die with the process (v17 lesson)."""
    global TE_LOADED_NAME
    src = f"{COMFY_DIR}/models/text_encoders/{TE_NAME}"
    dst = src.replace(".safetensors", "-fp8q.safetensors")
    if os.path.exists(dst) and os.path.getsize(dst) > 1_000_000_000:
        TE_LOADED_NAME = os.path.basename(dst)
        log(f"TE fp8q ready ({os.path.getsize(dst) / 2**30:.1f} GB)")
        return dst
    log("converting gemma TE bf16 → fp8q (ComfyUI mixed-ops format)…")
    conv_script = f"{BASE}/convert_te_fp8.py"
    with open(conv_script, "w") as f:
        f.write(CONVERT_TE_FP8_SRC
                .replace("__SRC__", src)
                .replace("__DST__", dst)
                .replace("__QRE__", QUANT_LIN_RE.pattern))
    r = subprocess.run([sys.executable, conv_script], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(dst):
        log(f"fp8q conversion failed rc={r.returncode}: {r.stdout[-300:]} {r.stderr[-600:]}")
        raise SystemExit("TE fp8q conversion failed")
    # the bf16 original is dead weight on the ephemeral disk — drop it
    try:
        os.remove(src)
    except Exception:
        pass
    TE_LOADED_NAME = os.path.basename(dst)
    log(f"TE fp8q written ({os.path.getsize(dst) / 2**30:.1f} GB), bf16 removed")
    return dst


CONVERT_TE_FP8_SRC = r'''
import json, os, re, struct
import numpy as np
import torch
from safetensors import safe_open

SRC = "__SRC__"
DST = "__DST__"
QRE = re.compile(r"__QRE__")
QJSON = json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")
DT_SIZE = {"U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "F16": 2, "BF16": 2,
           "F32": 4, "F64": 8, "I32": 4, "I16": 2, "U16": 2, "I64": 8}

with open(SRC, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))

# entries in original file order: (out_name, out_dtype, shape, src_key)
entries = []
for k, meta in sorted(((k, v) for k, v in hdr.items() if k != "__metadata__"),
                      key=lambda kv: kv[1]["data_offsets"][0]):
    dt, shape = meta["dtype"], list(meta["shape"])
    if dt in ("BF16", "F32", "F16") and len(shape) == 2 and QRE.match(k):
        base = k[: -len("weight")]
        entries.append((k, "F8_E4M3", shape, k))
        entries.append((base + "weight_scale", "F32", [], k))
        entries.append((base + "comfy_quant", "U8", [len(QJSON)], k))
    elif dt in ("BF16", "F32"):
        entries.append((k, "F16", shape, k))
    else:
        entries.append((k, dt, shape, k))

def numel(shape):
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)

# pass 1: compute data section layout (header is byte-size-stable upfront)
off, offsets = 0, {}
for name, dt, shape, _ in entries:
    nbytes = numel(shape) * DT_SIZE[dt]
    offsets[name] = (off, off + nbytes)
    off += nbytes
data_len = off

hdr_out = {"__metadata__": {"format": "pt"}}
for name, dt, shape, _ in entries:
    hdr_out[name] = {"dtype": dt, "shape": shape,
                     "data_offsets": list(offsets[name])}
hjson = json.dumps(hdr_out, separators=(",", ":")).encode("utf-8")
pad = (-(8 + len(hjson))) % 8
hjson += b" " * pad

# pass 2: stream tensor payloads in entry order
pend_scale = None
with open(DST, "wb") as out:
    out.write(struct.pack("<Q", len(hjson)))
    out.write(hjson)
    with safe_open(SRC, framework="pt", device="cpu") as f:
        for name, dt, shape, skey in entries:
            if dt == "U8" and name.endswith("comfy_quant"):
                out.write(QJSON)
            elif dt == "F8_E4M3":
                sl = f.get_slice(skey)
                rows, row_elems = shape[0], max(1, numel(shape[1:]))
                chunk = max(1, (64 << 20) // (row_elems * 4))
                amax = 0.0
                for s in range(0, rows, chunk):
                    amax = max(amax, sl[s: s + chunk].to(torch.float32).abs().amax().item())
                scale = max(amax / 448.0, 1e-12)
                pend_scale = scale
                for s in range(0, rows, chunk):
                    t = sl[s: s + chunk].to(torch.float32)
                    q = (t / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                    out.write(q.contiguous().view(torch.uint8).numpy().tobytes())
            elif dt == "F32" and name.endswith("weight_scale"):
                out.write(np.float32(pend_scale).tobytes())
                pend_scale = None
            elif dt == "F16":
                sl = f.get_slice(skey)
                rows, row_elems = shape[0], max(1, numel(shape[1:]))
                chunk = max(1, (64 << 20) // (row_elems * 2))
                for s in range(0, rows, chunk):
                    t = sl[s: s + chunk].to(torch.float16)
                    out.write(t.contiguous().numpy().tobytes())
            else:
                out.write(f.get_tensor(skey).contiguous().numpy().tobytes())

size = 8 + len(hjson) + data_len
assert os.path.getsize(DST) == size, f"size mismatch {os.path.getsize(DST)} != {size}"
nq = sum(1 for e in entries if e[1] == "F8_E4M3")
print(f"fp8q ok: {len(entries)} tensors ({nq} quantized), {size / 2**30:.2f} GB", flush=True)
'''


def log(msg: str) -> None:
    print(f"[vidgen {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


# ────────────────────────── setup: deps + models + comfy ──────────────────────────

def install_deps() -> None:
    log("installing deps…")
    sh("pip install -q boto3 'huggingface_hub[hf_transfer]' gguf av einops sentencepiece 2>&1 | tail -1 || true")


def gpu_count() -> int:
    try:
        r = subprocess.run("nvidia-smi --list-gpus | wc -l", shell=True,
                           capture_output=True, text=True, executable="/bin/bash", timeout=15)
        return max(1, int(r.stdout.strip() or "1"))
    except Exception:
        return 1


def hf_download(sub: str, name: str, dest_dir: str) -> str:
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 1_000_000:
        log(f"  cached: {name}")
        return dest
    from huggingface_hub import hf_hub_download
    os.makedirs(dest_dir, exist_ok=True)
    log(f"  downloading {sub}/{name} …")
    got = hf_hub_download(repo_id=HF_REPO, repo_type="model",
                          filename=f"{sub}/{name}", local_dir=BASE)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.abspath(got) != os.path.abspath(dest):
        os.replace(got, dest)
    log(f"  done: {name} ({os.path.getsize(dest) / 2**30:.2f} GB)")
    return dest


def setup_comfy() -> None:
    if not os.path.exists(COMFY_DIR):
        sh(f"git clone -q https://github.com/comfyanonymous/ComfyUI.git {COMFY_DIR}")
    sh(f"git -C {COMFY_DIR} fetch -q origin {COMFY_REV} 2>/dev/null || true")
    sh(f"git -C {COMFY_DIR} checkout -q --force {COMFY_REV}")

    plugin = f"{COMFY_DIR}/custom_nodes/ComfyUI-GGUF"
    if not os.path.exists(plugin):
        sh(f"git clone -q https://github.com/city96/ComfyUI-GGUF.git {plugin}")
    sh(f"git -C {plugin} fetch -q origin {GGUF_PLUGIN_REV} 2>/dev/null || true")
    sh(f"git -C {plugin} checkout -q --force {GGUF_PLUGIN_REV}")

    mgpu = f"{COMFY_DIR}/custom_nodes/ComfyUI-MultiGPU"
    if not os.path.exists(mgpu):
        sh(f"git clone -q https://github.com/pollockjj/ComfyUI-MultiGPU.git {mgpu}")
    sh(f"git -C {mgpu} fetch -q origin {MGPU_PLUGIN_REV} 2>/dev/null || true")
    sh(f"git -C {mgpu} checkout -q --force {MGPU_PLUGIN_REV}")

    log("installing ComfyUI requirements…")
    sh(f"pip install -q -r {COMFY_DIR}/requirements.txt 2>&1 | tail -1 || true")
    sh(f"pip install -q -r {plugin}/requirements.txt 2>&1 | tail -1 || true")

    log("fetching weights (unsloth mirror, ungated)…")
    hf_download(GGUF_SUB, GGUF_NAME, f"{COMFY_DIR}/models/diffusion_models")
    hf_download(TE_SUB, TE_NAME, f"{COMFY_DIR}/models/text_encoders")
    hf_download(VAE_SUB, VVAE_NAME, f"{COMFY_DIR}/models/vae")
    hf_download(VAE_SUB, AVAE_NAME, f"{COMFY_DIR}/models/vae")
    hf_download(UP_SUB, UP_NAME, f"{COMFY_DIR}/models/latent_upscale_models")


COMFY_PROCS = {}


def setup_swap() -> None:
    """Swapon is not permitted on Kaggle containers (v18 probe) — this is a
    no-op kept so the boot log records the attempt. The real RAM fix is the
    fp8q TE conversion."""
    try:
        sh("fallocate -l 12G /kaggle/temp/swapfile && chmod 600 /kaggle/temp/swapfile && "
           "mkswap -q /kaggle/temp/swapfile && swapon /kaggle/temp/swapfile 2>&1 || true")
        sh("free -m | tail -1 || true")
    except Exception as e:
        log(f"swap setup skipped: {e}")


def probe_te_keys() -> None:
    """Log the TE file's tensor names — informs a future GGUF-TE conversion
    (tokenizer asset tensors, proj head names, state-dict layout)."""
    try:
        from safetensors import safe_open
        p = f"{COMFY_DIR}/models/text_encoders/{TE_LOADED_NAME}"
        with safe_open(p, framework="pt", device="cpu") as f:
            keys = list(f.keys())
        log(f"TE tensors: {len(keys)}; sample: {keys[:6]}")
        tok = [k for k in keys if "token" in k.lower()]
        log(f"TE tokenizer-ish keys: {tok[:8]}")
    except Exception as e:
        log(f"TE probe failed: {e}")


def start_comfy_lane(tag: str, port: int, cuda_devices: str) -> None:
    """Launch one ComfyUI lane process and wait for its API."""
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    logfile = open(f"{BASE}/comfyui_{tag}.log", "a")
    # --disable-dynamic-vram + --cache-none: the fp8q TE (~half of 24.5 GiB)
    # full-loads to RAM, cache-none frees it before the 13 GiB GGUF DiT loads.
    # Never both at once, nothing staged to disk — keeps two lanes inside the
    # 29 GiB cgroup (the v4 OOM-killer lesson, re-checked for v7 dual-lane).
    cmd = [sys.executable, "main.py",
           "--port", str(port), "--listen", "127.0.0.1",
           "--lowvram", "--disable-dynamic-vram", "--cache-none", "--force-fp16",
           "--preview-method", "none", "--disable-auto-launch"]
    log(f"launching ComfyUI {tag} (port {port}, cuda {cuda_devices})")
    proc = subprocess.Popen(cmd, cwd=COMFY_DIR, env=env,
                            stdout=logfile, stderr=subprocess.STDOUT)
    COMFY_PROCS[tag] = (proc, port)
    api = f"http://127.0.0.1:{port}"
    deadline = time.time() + 25 * 60
    while time.time() < deadline:
        if proc.poll() is not None:
            sh(f"tail -40 {BASE}/comfyui_{tag}.log > {BASE}/comfyui_{tag}_tail.txt 2>/dev/null || true")
            raise SystemExit(f"ComfyUI {tag} exited early (rc={proc.returncode}) — see log tail")
        try:
            with urllib.request.urlopen(f"{api}/system_stats", timeout=5) as r:
                stats = json.loads(r.read())
            log(f"ComfyUI {tag} up: {stats.get('system', {}).get('comfyui_version', '?')}")
            return
        except Exception:
            time.sleep(5)
    raise SystemExit(f"ComfyUI {tag} did not come up in 25 minutes")


def object_info(api: str) -> dict:
    with urllib.request.urlopen(f"{api}/object_info", timeout=60) as r:
        return json.loads(r.read())


def api_post(api: str, path: str, payload: dict) -> dict:
    req = urllib.request.Request(f"{api}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


# ────────────────────────── workflow graph builder ──────────────────────────

def build_graph(mode: str, prompt: str, width: int, height: int, seed: int,
                ref_file: str, jid: str, stage: str = "two",
                duration_s: int = 5, split: bool = False) -> dict:
    """API-format graph cloned from the official Comfy-Org LTX-2.5 templates.
    stage='two' → 8 base steps at W/2×H/2 + latent 2x upsample + 3 refine steps.
    stage='one' → OOM fallback: 8 base steps directly at W×H.
    duration_s  → 5 (121 frames) or 15 (361 frames); 8n+1 constraint.
    split=True  → DisTorch2 combined mode: DiT resident on cuda:0 with PCIe
                  streaming from cuda:1, TE + VAEs resident on cuda:1
                  (15s jobs own both cards; v7 TASK-20261107-04)."""
    frames = duration_s * FPS + 1  # 8n+1
    g = {}
    n = [0]

    def node(cls, inputs=None, widgets=None):
        n[0] += 1
        nid = str(n[0])
        entry = {"class_type": cls, "inputs": {}}
        for k, v in (inputs or {}).items():
            entry["inputs"][k] = v
        for w in (widgets or []):
            entry["inputs"][w[0]] = w[1]
        g[nid] = entry
        return nid

    C = lambda nid, slot=0: [nid, slot]

    if split:
        clip = node("CLIPLoaderDisTorch2MultiGPU",
                    inputs={"compute_device": "cuda:1", "virtual_vram_gb": 14.0,
                            "donor_device": "cpu"},
                    widgets=[("clip_name", TE_LOADED_NAME), ("type", "ltxv")])
        unet = node("UnetLoaderGGUFDisTorch2MultiGPU",
                    inputs={"compute_device": "cuda:0", "virtual_vram_gb": 13.0,
                            "donor_device": "cuda:1"},
                    widgets=[("unet_name", GGUF_NAME)])
        vvae = node("VAELoaderDisTorch2MultiGPU",
                    inputs={"compute_device": "cuda:1", "virtual_vram_gb": 13.5,
                            "donor_device": "cpu"},
                    widgets=[("vae_name", VVAE_NAME)])
        avae = node("VAELoaderDisTorch2MultiGPU",
                    inputs={"compute_device": "cuda:1", "virtual_vram_gb": 13.5,
                            "donor_device": "cpu"},
                    widgets=[("vae_name", AVAE_NAME)])
    else:
        clip = node("CLIPLoader", widgets=[("clip_name", TE_LOADED_NAME), ("type", "ltxv"), ("device", "default")])
        unet = node("UnetLoaderGGUF", widgets=[("unet_name", GGUF_NAME)])
        vvae = node("VAELoader", widgets=[("vae_name", VVAE_NAME)])
        avae = node("VAELoader", widgets=[("vae_name", AVAE_NAME)])

    pos = node("CLIPTextEncode", inputs={"clip": C(clip)}, widgets=[("text", prompt)])
    neg = node("CLIPTextEncode", inputs={"clip": C(clip)}, widgets=[("text", NEGATIVE)])
    cond = node("LTXVConditioning", inputs={"positive": C(pos), "negative": C(neg)},
                widgets=[("frame_rate", float(FPS))])

    img_src = None
    if mode == "i2v":
        # ref image is pre-scaled in render_job (longer side → 1536, lanczos);
        # ResizeImageMaskNode's DynamicCombo API validation is brittle, so the
        # graph goes LoadImage → LTXVPreprocess directly.
        load = node("LoadImage", widgets=[("image", ref_file)])
        img_src = node("LTXVPreprocess", inputs={"image": C(load)}, widgets=[("img_compression", 18)])

    # stage 1: base pass at W/2 x H/2 (single-stage fallback: full W x H)
    bw, bh = (width // 2, height // 2) if stage == "two" else (width, height)
    vlat = node("EmptyLTXVLatentVideo", widgets=[("width", bw), ("height", bh),
                                                 ("length", frames), ("batch_size", 1)])
    alat = node("LTXVEmptyLatentAudio", inputs={"audio_vae": C(avae)},
                widgets=[("frames_number", frames), ("frame_rate", FPS), ("batch_size", 1)])
    av = node("LTXVConcatAVLatent", inputs={"video_latent": C(vlat), "audio_latent": C(alat)})

    latent_in = C(av)
    if mode == "i2v":
        # template wiring: the pin node takes the plain VIDEO latent (not the
        # packed AV latent — that one is a NestedTensor and has no .clone()),
        # and the concat consumes the pinned output.
        pin = node("LTXVImgToVideoInplace", inputs={"vae": C(vvae), "image": C(img_src), "latent": C(vlat)},
                   widgets=[("strength", 0.7), ("bypass", False)])
        av = node("LTXVConcatAVLatent", inputs={"video_latent": C(pin), "audio_latent": C(alat)})
        latent_in = C(av)

    noise1 = node("RandomNoise", widgets=[("noise_seed", int(seed)), ("control", "randomize")])
    ks1 = node("KSamplerSelect", widgets=[("sampler_name", "euler_ancestral")])
    sig1 = node("ManualSigmas", widgets=[("sigmas", SIGMAS_BASE)])
    g1 = node("LTXVDualCFGGuider", inputs={"model": C(unet), "positive": C(cond), "negative": C(cond)},
              widgets=[("video_cfg", 1.0), ("audio_cfg", 1.0)])
    samp1 = node("SamplerCustomAdvanced", inputs={"noise": C(noise1), "guider": C(g1),
                                                  "sampler": C(ks1), "sigmas": C(sig1),
                                                  "latent_image": latent_in})
    sep1 = node("LTXVSeparateAVLatent", inputs={"av_latent": C(samp1)})

    video_part = C(sep1, 0)
    if stage == "two":
        upm = node("LatentUpscaleModelLoader", widgets=[("model_name", UP_NAME)])
        up = node("LTXVLatentUpsampler", inputs={"samples": C(sep1, 0), "upscale_model": C(upm),
                                                 "vae": C(vvae)})
        video_part = C(up)
        if mode == "i2v":
            pin2 = node("LTXVImgToVideoInplace", inputs={"vae": C(vvae), "image": C(img_src),
                                                         "latent": C(up)},
                        widgets=[("strength", 1.0), ("bypass", False)])
            video_part = C(pin2)

    merged = node("LTXVConcatAVLatent", inputs={"video_latent": video_part, "audio_latent": C(sep1, 1)})

    noise2 = node("RandomNoise", widgets=[("noise_seed", 42), ("control", "fixed")])
    ks2 = node("KSamplerSelect", widgets=[("sampler_name", "euler_ancestral")])
    sig2 = node("ManualSigmas", widgets=[("sigmas", SIGMAS_REFINE if stage == "two" else SIGMAS_BASE)])
    g2 = node("LTXVDualCFGGuider", inputs={"model": C(unet), "positive": C(cond), "negative": C(cond)},
              widgets=[("video_cfg", 1.0), ("audio_cfg", 1.0)])
    samp2 = node("SamplerCustomAdvanced", inputs={"noise": C(noise2), "guider": C(g2),
                                                  "sampler": C(ks2), "sigmas": C(sig2),
                                                  "latent_image": C(merged)})
    sep2 = node("LTXVSeparateAVLatent", inputs={"av_latent": C(samp2)})
    dec = node("VAEDecodeTiled", inputs={"samples": C(sep2, 0), "vae": C(vvae)},
               widgets=[("tile_size", 512), ("overlap", 64), ("temporal_size", 64),
                        ("temporal_overlap", 16)])
    adec = node("LTXVAudioVAEDecode", inputs={"samples": C(sep2, 1), "audio_vae": C(avae)})
    vid = node("CreateVideo", inputs={"images": C(dec), "audio": C(adec)},
               widgets=[("fps", float(FPS)), ("bit_depth", 8)])
    node("SaveVideo", inputs={"video": C(vid)},
         widgets=[("filename_prefix", f"vidgen/{jid}"), ("format", "auto"), ("codec", "auto")])

    save_id = str(n[0])
    return {"prompt": g, "save_node": save_id}


# ────────────────────────── rendering ──────────────────────────

class Pinger:
    def __init__(self, ex: Exchange, jid: str, gpu: str) -> None:
        self.ex, self.jid, self.gpu = ex, jid, gpu
        self.progress = 10
        self._stop = False
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def set(self, p: int) -> None:
        self.progress = max(self.progress, min(int(p), 95))

    def _run(self) -> None:
        while not self._stop:
            try:
                self.ex.heartbeat(self.jid, "RENDERING", gpu=self.gpu,
                                  progress=self.progress, elapsed_s=time.time() - T0)
            except Exception:
                pass
            time.sleep(60)

    def stop(self) -> None:
        self._stop = True


def comfy_progress(tag: str) -> int:
    """Parse the latest sampler % from a lane's ComfyUI stdout log."""
    try:
        p = f"{BASE}/comfyui_{tag}.log"
        with open(p, "r", errors="ignore") as f:
            f.seek(max(0, os.path.getsize(p) - 40000))
            tail = f.read()
        pcts = re.findall(r"\s(\d{1,3})%\|", tail)
        return int(pcts[-1]) if pcts else 0
    except Exception:
        return 0


def debug_trail(ex: Exchange, jid: str, phase: str, extra: str = "") -> None:
    """Best-effort phase + ComfyUI log tail into R2 so a thrashed box is
    debuggable from the outside."""
    try:
        tails = []
        for tag, (proc, _port) in COMFY_PROCS.items():
            try:
                sz = os.path.getsize(f"{BASE}/comfyui_{tag}.log")
                with open(f"{BASE}/comfyui_{tag}.log", "r", errors="ignore") as f:
                    f.seek(max(0, sz - 1500))
                    tails.append(f"--- {tag} ---\n" + f.read())
            except Exception:
                pass
        ex.put_json(f"debug/{jid}.json", {
            "id": jid, "phase": phase, "extra": extra[:400],
            "comfy_tail": ("\n".join(tails))[-2200:], "at": time.time(),
        })
    except Exception:
        pass


def run_workflow(api: str, gf, pinger: Pinger, ex: Exchange, jid: str,
                 tag: str, timeout_min: int = 90) -> str:
    """POST /prompt, poll /history until the SaveVideo output exists. Returns mp4 path."""
    resp = api_post(api, "/prompt", {"prompt": gf["prompt"]})
    pid = resp["prompt_id"]
    log(f"[{tag}] queued prompt {pid}")
    deadline = time.time() + timeout_min * 60
    last_pct = -1
    while time.time() < deadline:
        time.sleep(8)
        # inline heartbeat: don't depend solely on the pinger thread
        try:
            ex.heartbeat(jid, "RENDERING", gpu=pinger.gpu,
                         progress=max(pinger.progress, 15 + int(comfy_progress(tag) * 0.75)),
                         elapsed_s=time.time() - T0)
        except Exception:
            pass
        try:
            with urllib.request.urlopen(f"{api}/history/{pid}", timeout=30) as r:
                hist = json.loads(r.read())
        except Exception:
            hist = {}
        entry = hist.get(pid)
        p = comfy_progress(tag)
        if p != last_pct:
            last_pct = p
            pinger.set(15 + int(p * 0.75))
            log(f"[{tag}] sampler {p}%")
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                msgs = []
                for m in (status.get("messages") or []):
                    if m and m[0] == "execution_error":
                        d = m[1] if len(m) > 1 else {}
                        msgs.append(f"{d.get('node_type')}: {d.get('exception_message')}")
                raise RuntimeError("execution error: " + " | ".join(msgs)[:500])
            out = entry.get("outputs", {}).get(gf["save_node"])
            if out:
                for k, v in out.items():
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict) and item.get("filename", "").endswith(".mp4"):
                                p = os.path.join(COMFY_DIR, "output", item.get("subfolder", ""),
                                                 item["filename"])
                                if os.path.exists(p):
                                    return p
                raise RuntimeError("SaveVideo output present but no mp4 found")
        if time.time() % 60 < 8:
            proc, _ = COMFY_PROCS[tag]
            if proc.poll() is not None:
                raise RuntimeError(f"ComfyUI {tag} died mid-render")
    raise RuntimeError(f"render timed out after {timeout_min} min")


def oom_evidence() -> str:
    """One-line OOM forensics for the fail status: kernel OOM-killer lines
    (when dmesg is readable) + RAM headroom, so a 'died mid-render' is
    diagnosable from the panel instead of churning blind sessions."""
    bits = []
    try:
        d = subprocess.run(
            "dmesg 2>/dev/null | grep -iE 'oom|killed process' | tail -2",
            shell=True, capture_output=True, text=True, executable="/bin/bash", timeout=10)
        if d.stdout.strip():
            bits.append("dmesg: " + d.stdout.strip().splitlines()[-1][:160])
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(("MemTotal", "MemAvailable")):
                    p = line.split()
                    bits.append(f"{p[0]}={int(p[1]) / 2**20:.1f}G")
    except Exception:
        pass
    return (" | ".join(bits))[:200]


def make_thumb(video_path: str, thumb_path: str) -> None:
    subprocess.run(f'ffmpeg -y -v error -ss 2 -i "{video_path}" -frames:v 1 -q:v 3 "{thumb_path}"',
                   shell=True, check=True, executable="/bin/bash")


def render_job(ex: Exchange, jid: str, spec: dict, lane: str) -> None:
    """Render one claimed job on `lane` ('lane0' hosts both single-card and
    split graphs; 'lane1' renders 5s single-card only)."""
    mode = spec.get("mode", "t2v")
    prompt = spec.get("prompt", "").strip()
    aspect = spec.get("aspect", "landscape")
    duration_s = int(spec.get("duration_s", 5))
    if duration_s not in DURATIONS:
        duration_s = 5
    split = duration_s == 15 and lane == "lane0"
    width, height = ASPECTS.get(aspect, ASPECTS["landscape"])
    seed = spec.get("seed")
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "little") % 4294967295
    gpu_label = f"t4:0+1/15s" if split else (f"t4:{'0' if lane == 'lane0' else '1'}/5s")

    api = f"http://127.0.0.1:{COMFY_PROCS[lane][1]}"

    ref_file = ""
    if mode == "i2v":
        os.makedirs(f"{COMFY_DIR}/input", exist_ok=True)
        raw_path = f"{BASE}/ref_{jid}.raw"
        got = ex.get_file(f"{ex.prefix}/jobs/{jid}/ref.jpg", raw_path) or \
              ex.get_file(f"{ex.prefix}/jobs/{jid}/ref.png", raw_path)
        if not got or os.path.getsize(raw_path) < 1000:
            raise RuntimeError("i2v reference image missing in the exchange")
        # pre-scale here (template used ResizeImageMaskNode: longer side → 1536, lanczos)
        from PIL import Image
        im = Image.open(raw_path).convert("RGB")
        w, h = im.size
        k = 1536 / max(w, h)
        if k < 1.0:
            im = im.resize((max(1, round(w * k)), max(1, round(h * k))), Image.LANCZOS)
        ref_file = f"ref_{jid}.jpg"
        im.save(f"{COMFY_DIR}/input/{ref_file}", quality=95)
        os.remove(raw_path)

    log(f"[{lane}] rendering {jid[:10]} mode={mode} dur={duration_s}s {width}x{height} "
        f"seed={seed} split={split}")
    pinger = Pinger(ex, jid, gpu_label)
    t0 = time.time()
    try:
        debug_trail(ex, jid, "submitting")
        gf = build_graph(mode, prompt, width, height, seed, ref_file, jid,
                         stage="two", duration_s=duration_s, split=split)
        stage_used = "two"
        try:
            video = run_workflow(api, gf, pinger, ex, jid, lane)
        except RuntimeError as e:
            low = str(e).lower()
            if "out of memory" not in low:
                raise
            # v7 fallback ladder: split two-stage OOM → single-stage full res;
            # then (15s only) dropped resolution bucket, per the approved plan.
            if split:
                log(f"split two-stage OOM → dropped-bucket retry ({str(e)[:120]})")
                ex.heartbeat(jid, "RENDERING", gpu=gpu_label, progress=12, elapsed_s=time.time() - T0)
                ow, oh = OOM_ASPECTS.get(aspect, OOM_ASPECTS["landscape"])
                gf = build_graph(mode, prompt, ow, oh, seed, ref_file, jid,
                                 stage="two", duration_s=duration_s, split=split)
                width, height = ow, oh
                stage_used = "two@dropped"
                video = run_workflow(api, gf, pinger, ex, jid, lane)
            else:
                log(f"two-stage OOM → single-stage fallback ({str(e)[:120]})")
                ex.heartbeat(jid, "RENDERING", gpu=gpu_label, progress=12, elapsed_s=time.time() - T0)
                gf = build_graph(mode, prompt, width, height, seed, ref_file, jid,
                                 stage="one", duration_s=duration_s, split=False)
                stage_used = "one"
                video = run_workflow(api, gf, pinger, ex, jid, lane)
        # thumb MUST end in .jpg — ffmpeg picks the container from the output
        # extension, so a .mp4 thumb name produced mp4 bytes inside thumb.jpg
        # (v21 bug: every poster in the gallery was an undecodable "image")
        thumb = os.path.join(os.path.dirname(video),
                             "thumb_" + os.path.basename(video).rsplit(".", 1)[0] + ".jpg")
        try:
            make_thumb(video, thumb)
        except Exception as te:
            log(f"thumb failed, publishing without poster: {str(te)[:150]}")
            thumb = ""
        debug_trail(ex, jid, "publishing")
        if thumb and os.path.exists(thumb):
            ex.publish_result(jid, video, thumb, gpu_label, time.time() - t0)
        else:
            # video alone is the deliverable — never lose a rendered clip to a
            # poster-frame hiccup (v20 lost both smoke renders to a bad
            # thumb path that pointed into a nonexistent subdirectory)
            ex.put_file(f"{ex.prefix}/results/{jid}/video.mp4", video)
            ex.heartbeat(jid, "DONE", gpu=gpu_label, progress=100, elapsed_s=time.time() - t0)
        log(f"[{lane}] DONE {jid[:10]} in {int(time.time() - t0)} s (stage={stage_used}, dur={duration_s}s)")
    except Exception as e:
        debug_trail(ex, jid, "failed", str(e))
        raise
    finally:
        pinger.stop()


# ────────────────────────── orchestrator main loop ──────────────────────────

def main() -> None:
    install_deps()
    setup_comfy()
    setup_swap()
    convert_te_fp8()
    probe_te_keys()

    ngpu = gpu_count()
    log(f"GPUs detected: {ngpu}")
    start_comfy_lane("lane0", 8188, "0,1" if ngpu > 1 else "0")
    if ngpu > 1:
        start_comfy_lane("lane1", 8189, "1")

    try:
        oi = object_info("http://127.0.0.1:8188")
        missing = [c for c in ("UnetLoaderGGUF", "LTXVConditioning", "LTXVDualCFGGuider",
                               "LTXVConcatAVLatent", "LTXVSeparateAVLatent", "LTXVAudioVAEDecode",
                               "LTXVEmptyLatentAudio", "LTXVLatentUpsampler", "LTXVImgToVideoInplace",
                               "LTXVPreprocess", "EmptyLTXVLatentVideo", "ManualSigmas",
                               "CreateVideo", "SaveVideo", "LatentUpscaleModelLoader",
                               "ResizeImageMaskNode") if c not in oi]
        if missing:
            raise SystemExit(f"node classes missing from ComfyUI: {missing}")
        if ngpu > 1:
            mgpu_missing = [c for c in ("UnetLoaderGGUFDisTorch2MultiGPU",
                                        "CLIPLoaderDisTorch2MultiGPU",
                                        "VAELoaderDisTorch2MultiGPU") if c not in oi]
            if mgpu_missing:
                raise SystemExit(f"DisTorch2 nodes missing: {mgpu_missing}")
            log("DisTorch2 split capability verified")
        ggufs = oi.get("UnetLoaderGGUF", {}).get("input", {}).get("required", {}).get("unet_name")
        if ggufs and GGUF_NAME not in ggufs[0]:
            log(f"WARNING: {GGUF_NAME} not in UnetLoaderGGUF list: {ggufs[0][:5]}")
        log(f"node check ok ({len(oi)} classes)")
    except SystemExit:
        raise
    except Exception as e:
        log(f"node check skipped: {e}")

    ex = Exchange()
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    log(f"box RAM: {int(line.split()[1]) / 2**20:.1f} GiB")
                    break
    except Exception:
        pass
    log(f"worker loop up — LTX-2.5 22B distilled UD-Q4_K_XL, lanes={min(ngpu, 2)}, "
        f"15s split={'on' if ngpu > 1 else 'off (single GPU)'}")
    MY_BOOT_TS = time.time()
    try:
        ex_boot = Exchange()
        ex_boot.put_json("boot_marker.json", {"at": MY_BOOT_TS, "pid": os.getpid(),
                                              "lanes": min(ngpu, 2), "split15": ngpu > 1})
    except Exception:
        MY_BOOT_TS = 0

    def superseded(ex_: Exchange) -> bool:
        """True when a NEWER session has announced itself — stale sessions must
        leave the exchange immediately instead of sniping fresh jobs with old
        code (the v12–v15 race)."""
        if not MY_BOOT_TS:
            return False
        try:
            m = ex_.get_json("boot_marker.json")
            return bool(m) and m.get("at", 0) > MY_BOOT_TS + 20
        except Exception:
            return False

    # per-lane dispatch queues + worker threads (single claimer: this loop)
    lanes = ["lane0"] + (["lane1"] if ngpu > 1 else [])
    queues: dict = {ln: queue.Queue() for ln in lanes}
    errors: queue.Queue = queue.Queue()
    fifteen_lock = threading.Lock()  # 15s owns the whole box while rendering

    def lane_worker(ln: str) -> None:
        while True:
            item = queues[ln].get()
            if item is None:
                return
            jid, spec = item
            try:
                render_job(ex, jid, spec, ln)
                errors.put(("ok", jid, ""))
            except Exception as e:
                msg = str(e)[:400]
                oom = oom_evidence()
                full = (msg + (" || " + oom if oom else ""))[:480]
                log(f"FAILED {jid[:10]}: {full}")
                try:
                    ex.fail(jid, full, gpu=ln)
                except Exception:
                    pass
                errors.put(("fail", jid, msg[:80]))
            finally:
                queues[ln].task_done()

    threads = {ln: threading.Thread(target=lane_worker, args=(ln,), daemon=True)
               for ln in lanes}
    for t in threads.values():
        t.start()

    def lanes_idle() -> bool:
        return all(q.empty() for q in queues.values())

    last_busy = time.time()
    fail_sig, fail_n = "", 0
    while time.time() - T0 < MAX_SESSION_MIN * 60:
        if superseded(ex):
            log("newer session announced itself — this worker stands down")
            return
        # surface failed renders to the circuit breaker
        while not errors.empty():
            kind, jid, sig = errors.get_nowait()
            if kind == "ok":
                fail_sig, fail_n = "", 0
                last_busy = time.time()
            else:
                fail_n = fail_n + 1 if sig == fail_sig else 1
                fail_sig = sig
                if fail_n >= 3:
                    debug_trail(ex, jid, "circuit-break",
                                f"3 consecutive same-signature failures: {sig}")
                    log("circuit breaker: 3 consecutive same-signature failures "
                        "— exiting instead of churning the queue")
                    raise SystemExit(3)
        try:
            jobs = ex.pending_jobs()
        except Exception as e:
            log(f"exchange poll failed: {e}")
            time.sleep(30)
            continue
        if not jobs:
            if time.time() - last_busy > IDLE_EXIT_MIN * 60:
                log(f"queue empty for {IDLE_EXIT_MIN} min — exiting")
                return
            time.sleep(15)
            continue
        # 15s jobs own the box: dispatch only when every lane is idle
        fifteen = next((j for j in jobs if int(j.get("duration_s", 5)) == 15), None)
        if fifteen is not None:
            if lanes_idle() and fifteen_lock.acquire(blocking=False):
                try:
                    if superseded(ex):
                        log("newer session announced itself — not claiming the 15s job")
                        break
                    claimed = ex.claim(fifteen["id"], fifteen.get("_key", ""))
                    last_busy = time.time()
                    if claimed is not None:
                        log(f"15s job {fifteen['id'][:10]} — claiming the whole box")
                        queues["lane0"].put((fifteen["id"], claimed))
                        queues["lane0"].join()   # box is single-purpose until done
                finally:
                    fifteen_lock.release()
                continue
            time.sleep(15)
            continue
        # regular 5s jobs → any lane with an empty queue (round-robin-ish)
        for spec in jobs:
            if superseded(ex):
                log("newer session announced itself — not claiming further jobs")
                return
            if int(spec.get("duration_s", 5)) != 5:
                continue  # handled by the 15s branch on a later pass
            free = next((ln for ln in lanes if queues[ln].empty()), None)
            if free is None:
                break
            claimed = ex.claim(spec["id"], spec.get("_key", ""))
            if claimed is None:
                continue
            last_busy = time.time()
            log(f"dispatching {spec['id'][:10]} → {free}")
            queues[free].put((spec["id"], claimed))

    log("session cap reached — exiting")


if __name__ == "__main__":
    main()
