# Atelier VidGen — Architecture & Decisions

## Why this shape

House pattern (inherited from Atelier imagegen):
GitHub repo → edge-served panel → object-storage exchange → CLI-first ops →
vault-managed creds → MemO/Notion bookkeeping. VidGen adds one new organ: a
**Kaggle kernel as the GPU worker**, which forces an async, queue-based UX
instead of a live websocket pipe.

## The exchange (R2, prefix `vidgen/`)

```
jobs/<id>/job.json       pending spec — existence == queued
jobs/<id>/ref.(jpg|png)  i2v reference frame
status/<id>.json         heartbeat {status, gpu, progress, elapsed_s, updated_at}
results/<id>/video.mp4   finished clip (h264 + aac, faststart)
results/<id>/thumb.jpg   poster frame (frame 24)
```

- **Claim = delete `job.json`.** S3 delete semantics are the atomic primitive:
  with dual-T4 workers polling the same queue, only one delete succeeds, so
  the winner renders and losers skip. No lock service needed.
- **Heartbeats every phase transition** (15% → 40% → 80% → DONE/FAILED).
- **Reaper:** bridge requeues any RENDERING job whose heartbeat is >15 min
  stale (worker session died — routine on Kaggle).

## Dual-T4 policy (sir's directive: "split resources in 2 cards if best")

**Chosen: one worker process per card** (`CUDA_VISIBLE_DEVICES=0/1`, two
processes, each with its own pipeline instance) → **2× queue throughput**,
zero coordination cost, graceful degradation to single-GPU.

**Rejected: model/pipeline parallelism across the two cards.** A 3B distilled
model fits one 16 GB card with module offload; splitting text-encoder/DiT/VAE
across a PCIe link adds sync stalls per step for a model that doesn't need
the headroom. Efficiency at this size = parallel jobs, not a wider machine.

## Failure modes & answers

| Failure | Answer |
|---|---|
| Worker session dies mid-render | stale-claim reaper requeues after 15 min |
| Bridge dies | stateless; restart resumes from DB + exchange truth |
| Panel shows stale status | 4 s polling; DB is only a cache of exchange truth |
| Quota burn while idle | worker exits after 10 idle min, hard cap 5 h/session |
| Double-render race | delete-as-claim (S3 atomicity) |
| i2v ref image loss | ref uploaded to exchange at submit; worker re-fetches |

## Quota math

30 GPU-h/wk (B only). At 6–15 min/clip → **120–250 clips/week**. Session
strategy: run the worker when a queue exists (kernels push / resume), stop on
idle. First-session extra cost: model download (~35 GB) ≈ 10–20 min once per
session; weights cached in session disk across jobs.

## Security

- R2 creds never leave server-side: bridge/kernel read them from
  chmod-600 env files (`scripts/vidgen_runtime.env`) or the private Kaggle
  dataset `xion678/vidgen-config`. The panel never touches them.
- Kaggle KGAT token for account B lives in `scripts/kaggle_b_token.env`
  (vault-restored, never logged).

## Version pins / model identity

- Model: `kandinskylab/Kandinsky-6.0-Lite-distill-5s-Diffusers`
- Sampler: PiFlow distilled, 10 steps, guidance 1.0 (per official repo)
- Output: 121 frames @ 24 fps = 5.0 s; portrait 480×864, landscape 864×480
- dtype on T4 (sm75): fp16 (no bf16 support on Turing)
