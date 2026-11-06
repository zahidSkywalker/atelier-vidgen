/**
 * Atelier VidGen edge worker — one stable URL, four jobs:
 *   1. serve the panel from Static Assets (0 CPU for asset hits)
 *   2. /api/jobs POST   → write job spec (+ i2v ref) straight into the R2 exchange
 *      /api/jobs GET    → queue + gallery state (status/*.json + edge index)
 *   3. /api/media/*     → mp4 / thumb streamed from the R2 binding
 *   4. /api/wake        → push the GPU worker kernel to Kaggle (SaveKernel API),
 *                         boot code + metadata read from R2 vidgen/boot/*
 *
 * Every /api/* call (except /api/knock) requires the panel PIN — sent as
 * header X-Vidgen-Pin, cookie vidgen_pin, or ?pin= (media tags can't set headers).
 * Layout mirrors the worker exchange protocol: vidgen/jobs|status|results.
 */

const PIN_COOKIE = "vidgen_pin";
const PREFIX = "vidgen";

const json = (data, status = 200, extra = {}) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...extra },
  });

function timingSafeEqual(a, b) {
  const len = Math.max(a.length, b.length);
  let diff = a.length === b.length ? 0 : 1;
  for (let i = 0; i < len; i++) {
    diff |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  }
  return diff === 0;
}

async function getState(env) {
  const obj = await env.CTRL.get(`${PREFIX}/state.json`);
  if (!obj) return {};
  try { return await obj.json(); } catch { return {}; }
}

async function getPin(request, url) {
  let pin = request.headers.get("X-Vidgen-Pin") || "";
  if (!pin) {
    const cookie = request.headers.get("Cookie") || "";
    const m = cookie.match(new RegExp(`(?:^|;\\s*)${PIN_COOKIE}=([^;]*)`));
    if (m) pin = decodeURIComponent(m[1]);
  }
  if (!pin && url) pin = url.searchParams.get("pin") || "";
  return pin;
}

async function requirePin(request, env, url) {
  const state = await getState(env);
  const expected = state.panel_pin || "";
  if (!expected) return { state, err: json({ error: "not provisioned — state.json has no panel_pin" }, 503) };
  const got = await getPin(request, url);
  if (!timingSafeEqual(got, expected)) return { state, err: json({ error: "bad PIN" }, 401) };
  return { state, err: null };
}

/* ---------- exchange helpers ---------- */

async function listStatuses(env, limit = 80) {
  const list = await env.CTRL.list({ prefix: `${PREFIX}/status/`, limit: 1000 });
  const newest = list.objects
    .filter((o) => o.key.endsWith(".json"))
    .sort((a, b) => b.uploaded - a.uploaded)
    .slice(0, limit);
  const out = [];
  for (const obj of newest) {
    try {
      const st = await (await env.CTRL.get(obj.key)).json();
      out.push({
        id: st.id, status: st.status, gpu: st.gpu || null, progress: st.progress || 0,
        elapsed_s: st.elapsed_s || 0, error: st.error || "",
        updated_at: Math.round(obj.uploaded / 1000),
      });
    } catch { /* skip corrupt */ }
  }
  return out;
}

async function getIndex(env, ids) {
  const out = {};
  await Promise.all(ids.map(async (id) => {
    try {
      const obj = await env.CTRL.get(`${PREFIX}/edge_index/${id}.json`);
      if (obj) out[id] = await obj.json();
    } catch { /* skip */ }
  }));
  return out;
}

/* ---------- WAKE: push the GPU kernel to Kaggle ---------- */

const KAGGLE_SAVE_URL = "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel";

async function wakeKernel(request, env, url) {
  const { state, err } = await requirePin(request, env, url);
  if (err) return err;

  // a wake was issued <10 min ago? the lane is already booting — don't stack sessions
  if (state.wake_ts && Date.now() / 1000 - state.wake_ts < 600) {
    return json({ ok: true, already_waking: true, detail: "a T4 session is already booting — riding it out" });
  }
  if (!env.KAGGLE_API_TOKEN) {
    return json({ error: "wake not provisioned — KAGGLE_API_TOKEN secret missing on the worker" }, 503);
  }

  const [codeObj, metaObj] = await Promise.all([
    env.CTRL.get(`${PREFIX}/boot/vidgen_kernel.py`),
    env.CTRL.get(`${PREFIX}/boot/kernel-metadata.json`),
  ]);
  if (!codeObj || !metaObj) {
    return json({ error: "boot assets missing in R2 (vidgen/boot/*)" }, 500);
  }
  const code = await codeObj.text();
  const meta = await metaObj.json();

  await env.CTRL.put(`${PREFIX}/state.json`, JSON.stringify({
    ...state, wake_ts: Math.round(Date.now() / 1000),
  }));

  const body = {
    slug: meta.id,
    newTitle: meta.title,
    text: code,
    language: meta.language || "python",
    kernelType: meta.kernel_type || "script",
    isPrivate: meta.is_private !== false,
    enableGpu: !!meta.enable_gpu,
    enableTpu: !!meta.enable_tpu,
    enableInternet: meta.enable_internet !== false,
    datasetDataSources: meta.dataset_sources || [],
    kernelDataSources: meta.kernel_sources || [],
    modelDataSources: meta.model_sources || [],
  };
  let resp;
  try {
    resp = await fetch(KAGGLE_SAVE_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${env.KAGGLE_API_TOKEN}`,
        "Content-Type": "application/json",
        "User-Agent": "vidgen-edge/1.0",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000),
    });
  } catch (e) {
    return json({ error: "kaggle api unreachable", detail: String(e).slice(0, 200) }, 502);
  }
  let data = {};
  try { data = await resp.json(); } catch { /* non-json */ }
  if (!resp.ok || (data && data.error)) {
    const detail = data?.error?.message || data?.error || `kaggle http ${resp.status}`;
    return json({ error: "kaggle rejected the push", detail: String(detail).slice(0, 300) }, 502);
  }
  return json({
    ok: true,
    version: data.versionNumber ?? null,
    url: data.url ?? null,
    detail: "T4s booting — worker claims jobs in 3–8 min, first clip lands ~15–25 min after that",
  });
}

/* ---------- router ---------- */

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    if (!path.startsWith("/api/")) {
      return env.ASSETS.fetch(request);
    }

    try {
      // PIN knock → cookie (the only unguarded route)
      if (path === "/api/knock" && request.method === "POST") {
        const state = await getState(env);
        const expected = state.panel_pin || "";
        let got = "";
        try { got = (await request.json()).pin || ""; } catch { /* bad body */ }
        if (!expected) return json({ error: "not provisioned" }, 503);
        if (!timingSafeEqual(got, expected)) return json({ error: "bad PIN" }, 401);
        const resp = json({ ok: true });
        resp.headers.append("set-cookie",
          `${PIN_COOKIE}=${encodeURIComponent(got)}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000`);
        return resp;
      }

      // queue + gallery
      if (path === "/api/jobs" && request.method === "GET") {
        const { state, err } = await requirePin(request, env, url);
        if (err) return err;
        const statuses = await listStatuses(env);
        const index = await getIndex(env, statuses.map((s) => s.id));
        const jobs = statuses.map((s) => ({
          ...s,
          prompt: index[s.id]?.prompt || "",
          mode: index[s.id]?.mode || null,
          aspect: index[s.id]?.aspect || null,
          created_at: index[s.id]?.created_at || null,
          media: s.status === "DONE"
            ? { video: `/api/media/${PREFIX}/results/${s.id}/video.mp4`, thumb: `/api/media/${PREFIX}/results/${s.id}/thumb.jpg` }
            : null,
        }));
        const wake_age = state.wake_ts ? Math.round(Date.now() / 1000) - state.wake_ts : null;
        return json({
          jobs,
          lane: {
            waking_recently: wake_age !== null && wake_age < 600,
            wake_ts: state.wake_ts || null,
            pending: jobs.filter((j) => j.status === "QUEUED").length,
            rendering: jobs.filter((j) => j.status === "RENDERING").length,
          },
        });
      }

      // submit
      if (path === "/api/jobs" && request.method === "POST") {
        const { err } = await requirePin(request, env, url);
        if (err) return err;
        let body;
        try { body = await request.json(); } catch { return json({ error: "bad json" }, 400); }
        const mode = body.mode === "i2v" ? "i2v" : body.mode === "t2v" ? "t2v" : null;
        if (!mode) return json({ error: "mode must be t2v or i2v" }, 400);
        const prompt = String(body.prompt || "").trim();
        if (!prompt) return json({ error: "prompt is required" }, 400);
        if (prompt.length > 1200) return json({ error: "prompt too long (max 1200)" }, 400);
        const aspect = body.aspect === "landscape" ? "landscape" : "portrait";
        let seed = null;
        if (body.seed !== undefined && body.seed !== null && String(body.seed) !== "") {
          const n = Number(body.seed);
          if (!Number.isInteger(n) || n < 0 || n > 4294967295) return json({ error: "seed must be an integer 0..4294967295" }, 400);
          seed = n;
        }
        const audio = body.audio === false ? false : true;

        const id = crypto.randomUUID().replace(/-/g, "").slice(0, 24);
        const spec = {
          id, mode, prompt, aspect, seed, audio, super_res: false,
          created_at: new Date().toISOString(),
        };
        await env.CTRL.put(`${PREFIX}/jobs/${id}/job.json`, JSON.stringify(spec), {
          httpMetadata: { contentType: "application/json" },
        });

        if (mode === "i2v") {
          const refB64 = String(body.ref_b64 || "");
          const m = /^data:image\/(png|jpeg|jpg);base64,(.+)$/i.exec(refB64) ||
                    (/^[A-Za-z0-9+/=\s]+$/.test(refB64) && refB64.length > 100 ? [null, body.ref_mime || "jpeg", refB64] : null);
          if (!m) return json({ error: "i2v requires ref_b64 (data:image/...;base64,...)" }, 400);
          const ext = m[1].toLowerCase() === "png" ? "png" : "jpg";
          const bytes = Uint8Array.from(atob(m[2].replace(/\s/g, "")), (c) => c.charCodeAt(0));
          if (bytes.length > 12 * 1024 * 1024) return json({ error: "reference image too large (max 12 MB)" }, 400);
          await env.CTRL.put(`${PREFIX}/jobs/${id}/ref.${ext}`, bytes, {
            httpMetadata: { contentType: ext === "png" ? "image/png" : "image/jpeg" },
          });
        }

        await env.CTRL.put(`${PREFIX}/edge_index/${id}.json`, JSON.stringify({
          prompt, mode, aspect, created_at: spec.created_at,
        }), { httpMetadata: { contentType: "application/json" } });

        return json({ ok: true, id, detail: "queued on the exchange" });
      }

      // media streaming
      const mediaMatch = path.match(/^\/api\/media\/(.+)$/);
      if (mediaMatch && request.method === "GET") {
        const { err } = await requirePin(request, env, url);
        if (err) return err;
        const key = decodeURIComponent(mediaMatch[1]);
        if (!key.startsWith(`${PREFIX}/results/`)) return json({ error: "not a media key" }, 400);
        if (!/\.(mp4|jpg|jpeg|png)$/.test(key)) return json({ error: "bad media type" }, 400);
        const obj = await env.CTRL.get(key);
        if (!obj) return json({ error: "not found" }, 404);
        const ct = key.endsWith(".mp4") ? "video/mp4"
          : key.endsWith(".png") ? "image/png" : "image/jpeg";
        return new Response(obj.body, {
          headers: { "content-type": ct, "cache-control": "private, max-age=86400" },
        });
      }

      // WAKE
      if (path === "/api/wake" && request.method === "POST") {
        return wakeKernel(request, env, url);
      }

      return json({ error: "unknown api route" }, 404);
    } catch (e) {
      return json({ error: "worker exception", detail: String(e).slice(0, 200) }, 500);
    }
  },
};
