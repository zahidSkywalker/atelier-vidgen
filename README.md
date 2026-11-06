# Atelier VidGen 🎬

Cinematic video-generation webui — the video sibling of **Atelier**.
Text-to-video and image-to-video, powered by **Kandinsky 6.0 Video Lite (3B, MIT)**
on a single free compute lane: **Kaggle account B (xion678, 2× Tesla T4, 30 GPU-h/week)**.

- 5-second clips, 480×864 portrait / 864×480 landscape, 24 fps
- **Synchronized 44 kHz audio generated in the same pass** (Kandinsky native)
- 10-step PiFlow distilled render, guidance 1.0
- Dark, responsive panel — phone → desktop, one codebase (Next.js 16 + shadcn/ui)
- Async job lane: submit in the browser, collect the clip from the gallery

## Architecture

```
┌─────────────────────────┐        ┌──────────────────────────────┐
│  PANEL (Next.js)        │  jobs  │  EXCHANGE (R2, prefix vidgen)│
│  T2V tab · I2V tab      │───────►│  jobs/<id>/job.json + ref    │
│  queue · gallery        │        │  status/<id>.json heartbeat  │
│  (Prisma/SQLite store)  │◄───────│  results/<id>/video.mp4      │
└───────────┬─────────────┘  poll  └──────────────┬───────────────┘
            │ bridge (scripts/vidgen_bridge.py)   │ poll/claim/upload
            └─────────────────────────────────────┴───────────────┐
                                                    ┌─────────────▼─────────────┐
                                                    │ WORKER — Kaggle B kernel  │
                                                    │ Kandinsky 6.0 Lite        │
                                                    │ dual-T4: 1 process/card   │
                                                    │ idle-exit + session cap   │
                                                    └───────────────────────────┘
```

**Claim protocol:** deleting `jobs/<id>/job.json` claims the job — S3 semantics
guarantee exactly one deleter wins, so N workers never double-render.
**Stale-claim reaper:** heartbeats older than 15 min are requeued by the bridge
(survives worker-session death — the #1 operational failure on Kaggle).

## Repo layout

| Path | What |
|---|---|
| `panel/` | Next.js 16 + Tailwind 4 + shadcn/ui webui (T2V/I2V, queue, gallery) |
| `worker/vidgen_kernel.py` | Kaggle GPU worker kernel (dual-T4, quota-disciplined) |
| `worker/kernel-metadata.json` | Kaggle kernel spec (push via `kaggle kernels push -p worker`) |
| `scripts/vidgen_exchange.py` | R2 exchange protocol (shared by bridge + workers) |
| `scripts/vidgen_bridge.py` | panel-DB ↔ R2 sync loop + reaper |
| `scripts/vidgen_mock_worker.py` | CPU mock worker (ffmpeg) — full E2E without GPU |
| `docs/architecture.md` | design decisions, quota math, failure modes |

## Quota math (Kaggle B)

- 30 GPU-h/week → ~4.3 h/day; est. **120–250 clips/week** at 6–15 min/clip
- Worker exits after `IDLE_EXIT_MIN` (10) idle minutes; hard cap `MAX_SESSION_MIN` (300)
- Dual-T4 mode doubles throughput inside one session
- TPU (20 h/wk) sits unused — Kandinsky is CUDA-only

## Panel dev quickstart

```bash
cd panel
npm install && npx prisma db push && npm run dev   # http://localhost:3000
# E2E without GPU:
python3 ../scripts/vidgen_bridge.py &              # DB ↔ R2
python3 ../scripts/vidgen_mock_worker.py &         # claims + renders mock clips
```

The real worker needs `xion678/vidgen-config` (private dataset: R2 creds +
exchange module) attached to the kernel.

## Status

- [x] M0 — panel shell, T2V/I2V, responsive, mock jobs
- [x] M1 — R2 exchange + bridge + mock worker, end-to-end green (t2v + i2v + orphan requeue)
- [x] GPU attach verification on Kaggle B — **2× Tesla T4 confirmed** (0.01 h burned)
- [ ] M2 — first real Kandinsky render session on Kaggle B
- [ ] M3 — quota meter in panel (live from Kaggle API)
- [ ] M4 — Full-HD super-res toggle, polish

---
Authored by Diana for Dr. Zod · Zahidul Islam <220056915+zahidSkywalker@users.noreply.github.com>
