#!/usr/bin/env python3
"""Atelier VidGen bridge — syncs panel DB (Prisma sqlite) ↔ R2 exchange.

Panel writes QUEUED jobs to sqlite; bridge pushes them to R2 for the GPU lane
(Kaggle kernel or mock worker), pulls heartbeats back, and downloads finished
clips into public/results so the panel can play them directly.

Run: python3 scripts/vidgen_bridge.py   (idempotent, safe with pidfile)
"""
import json
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vidgen_exchange import Exchange  # noqa: E402

ROOT = "/home/z/my-project"
DB = os.path.join(ROOT, "db", "custom.db")
PUBLIC = os.path.join(ROOT, "public")
RESULTS_DIR = os.path.join(PUBLIC, "results")
PIDFILE = "/tmp/vidgen_bridge.pid"
STATE = "/tmp/vidgen_bridge_state.json"
POLL_S = 4


def log(msg: str) -> None:
    print(f"[bridge {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def db_rows(con: sqlite3.Connection, sql: str, args=()):
    con.row_factory = sqlite3.Row
    return con.execute(sql, args).fetchall()


def main() -> None:
    if os.path.exists(PIDFILE):
        try:
            pid = int(open(PIDFILE).read().strip())
            os.kill(pid, 0)
            print(f"bridge already running (pid {pid})")
            return
        except (ProcessLookupError, ValueError):
            pass
    open(PIDFILE, "w").write(str(os.getpid()))

    ex = Exchange()
    con = sqlite3.connect(DB, timeout=15)
    con.row_factory = sqlite3.Row
    os.makedirs(RESULTS_DIR, exist_ok=True)
    log("bridge up — DB ↔ R2 exchange")

    while True:
        try:
            # 1) push QUEUED jobs to R2 (state file dedupes resubmits)
            queued = con.execute(
                "SELECT * FROM Job WHERE status='QUEUED' ORDER BY createdAt ASC"
            ).fetchall()
            state = {}
            if os.path.exists(STATE):
                try:
                    state = json.loads(open(STATE).read())
                except Exception:
                    state = {}
            for row in queued:
                jid = row["id"]
                if state.get(jid) == "submitted":
                    continue
                spec = {
                    "id": jid,
                    "mode": row["mode"],
                    "prompt": row["prompt"],
                    "aspect": row["aspect"],
                    "seed": row["seed"],
                    "audio": bool(row["audio"]),
                    "super_res": bool(row["superRes"]),
                    "created_at": row["createdAt"],
                }
                ref_abs = None
                if row["mode"] == "i2v" and row["refImage"]:
                    p = os.path.join(PUBLIC, row["refImage"].lstrip("/"))
                    if os.path.exists(p):
                        ref_abs = p
                try:
                    ex.submit_job(spec, ref_abs)
                    state[jid] = "submitted"
                    json.dump(state, open(STATE, "w"))
                    log(f"submitted {jid[:10]} → exchange ({row['mode']})")
                except Exception as e:
                    log(f"submit {jid[:10]} failed: {e}")

            # 2) pull heartbeats for live jobs
            live = con.execute(
                "SELECT * FROM Job WHERE status IN ('QUEUED','RENDERING')"
            ).fetchall()
            for row in live:
                jid = row["id"]
                st = ex.status(jid)
                if not st:
                    continue
                rstatus = st.get("status")
                if rstatus == "RENDERING" and row["status"] != "RENDERING":
                    con.execute(
                        "UPDATE Job SET status='RENDERING', startedAt=?, gpu=? WHERE id=?",
                        (time.strftime("%Y-%m-%d %H:%M:%S"), st.get("gpu") or "lane", jid),
                    )
                    con.commit()
                    log(f"{jid[:10]} rendering on {st.get('gpu', 'lane')}")
                elif rstatus == "RENDERING":
                    # stale-claim reaper: heartbeat older than 15 min = worker died
                    age = time.time() - float(st.get("updated_at", 0))
                    if age > 900:
                        log(f"{jid[:10]} stale claim ({age:.0f}s) — requeueing")
                        try:
                            ex.delete(f"{ex.prefix}/status/{jid}.json")
                            spec = {
                                "id": jid,
                                "mode": row["mode"],
                                "prompt": row["prompt"],
                                "aspect": row["aspect"],
                                "seed": row["seed"],
                                "audio": bool(row["audio"]),
                                "super_res": bool(row["superRes"]),
                                "created_at": row["createdAt"],
                            }
                            ref_abs = None
                            if row["mode"] == "i2v" and row["refImage"]:
                                p = os.path.join(PUBLIC, row["refImage"].lstrip("/"))
                                if os.path.exists(p):
                                    ref_abs = p
                            ex.submit_job(spec, ref_abs)
                            con.execute(
                                "UPDATE Job SET status='QUEUED', gpu=NULL, progress=0 WHERE id=?",
                                (jid,),
                            )
                            con.commit()
                        except Exception as e:
                            log(f"requeue {jid[:10]} failed: {e}")
                        continue
                    con.execute(
                        "UPDATE Job SET elapsedMs=?, progress=? WHERE id=?",
                        (int(st.get("elapsed_s", 0) * 1000), int(st.get("progress", 0)), jid),
                    )
                    con.commit()
                elif rstatus == "DONE":
                    video_key = f"{ex.prefix}/results/{jid}/video.mp4"
                    thumb_key = f"{ex.prefix}/results/{jid}/thumb.jpg"
                    vpath = os.path.join(RESULTS_DIR, f"{jid}.mp4")
                    tpath = os.path.join(RESULTS_DIR, f"{jid}_thumb.jpg")
                    ok_v = ex.get_file(video_key, vpath)
                    ok_t = ex.get_file(thumb_key, tpath)
                    if ok_v:
                        con.execute(
                            "UPDATE Job SET status='DONE', videoPath=?, thumbPath=?, "
                            "finishedAt=?, elapsedMs=?, progress=100 WHERE id=?",
                            (
                                f"/results/{jid}.mp4",
                                f"/results/{jid}_thumb.jpg" if ok_t else None,
                                time.strftime("%Y-%m-%d %H:%M:%S"),
                                int(st.get("elapsed_s", 0) * 1000),
                                jid,
                            ),
                        )
                        con.commit()
                        log(f"{jid[:10]} DONE ({st.get('elapsed_s', 0)}s on {st.get('gpu', 'lane')})")
                    else:
                        con.execute(
                            "UPDATE Job SET status='FAILED', error='result missing on exchange', "
                            "finishedAt=? WHERE id=?",
                            (time.strftime("%Y-%m-%d %H:%M:%S"), jid),
                        )
                        con.commit()
                elif rstatus == "FAILED":
                    con.execute(
                        "UPDATE Job SET status='FAILED', error=?, finishedAt=? WHERE id=?",
                        (st.get("error") or "render failed", time.strftime("%Y-%m-%d %H:%M:%S"), jid),
                    )
                    con.commit()
                    log(f"{jid[:10]} FAILED: {st.get('error', '')[:80]}")
        except Exception as e:
            log(f"loop error: {e}")
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
