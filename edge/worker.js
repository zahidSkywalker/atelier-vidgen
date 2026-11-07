// Atelier VidGen — edge worker.
// Static panel (ASSETS) + PIN gate + R2 exchange reads + Kaggle SaveKernel wake.
// Engine: LTX-2.5 22B distilled (Unsloth UD-Q4_K_XL GGUF) on Kaggle 2×T4, ComfyUI-GGUF.

const PIN_COOKIE = "vidgen_pin";
const PREFIX = "vidgen";
const ENGINE = "LTX-2.5 22B distilled · UD-Q4_K_XL";
const KAGGLE_SAVE_URL = "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel";
const WAKE_COOLDOWN_S = 600;   // treat a wake as "active" for 10 min
const MEDIA_VER = 2;           // bump when result assets are regenerated — busts browser cache

const json = (data, status = 200, extra = {}) =>
  new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8", "cache-control": "no-store", ...extra },
  });

function timingSafeEqual(a, b) {
  const len = Math.max(a.length, b.length);
  let diff = a.length === b.length ? 0 : 1;
  for (let i = 0; i < len; i++) diff |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  return diff === 0;
}

async function getState(env) {
  const obj = await env.CTRL.get(`${PREFIX}/state.json`);
  if (!obj) return {};
  try { return await obj.json(); } catch { return {}; }
}

function getPin(request, url) {
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
  const got = getPin(request, url);
  if (!timingSafeEqual(got, expected)) return { state, err: json({ error: "bad PIN" }, 401) };
  return { state, err: null };
}

async function listStatuses(env, limit = 80) {
  const list = await env.CTRL.list({ prefix: `${PREFIX}/status/`, limit: 1000 });
  const newest = list.objects.filter((o) => o.key.endsWith(".json")).sort((a, b) => b.uploaded - a.uploaded).slice(0, limit);
  const out = [];
  for (const obj of newest) {
    try {
      const st = await (await env.CTRL.get(obj.key)).json();
      out.push({
        id: st.id,
        status: st.status,
        gpu: st.gpu || null,
        progress: st.progress || 0,
        elapsed_s: st.elapsed_s || 0,
        error: st.error || "",
        updated_at: Math.round(obj.uploaded / 1e3),
      });
    } catch {}
  }
  return out;
}

async function getIndex(env, ids) {
  const out = {};
  await Promise.all(ids.map(async (id) => {
    try {
      const obj = await env.CTRL.get(`${PREFIX}/edge_index/${id}.json`);
      if (obj) out[id] = await obj.json();
    } catch {}
  }));
  return out;
}

// NOTE: the kernel writes boot_marker.json (and debug/*.json) at BUCKET ROOT —
// its raw R2 helpers don't apply the vidgen/ prefix. Read it where it lands.
async function getBootMarker(env) {
  try {
    const obj = await env.CTRL.get("boot_marker.json");
    if (!obj) return null;
    const m = await obj.json();
    return m && m.at ? m : null;
  } catch { return null; }
}

async function wakeKernel(request, env, url) {
  const { state, err } = await requirePin(request, env, url);
  if (err) return err;

  if (state.wake_ts && Date.now() / 1e3 - state.wake_ts < WAKE_COOLDOWN_S) {
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

  await env.CTRL.put(`${PREFIX}/state.json`, JSON.stringify({ ...state, wake_ts: Math.round(Date.now() / 1e3) }));

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
        "User-Agent": "vidgen-edge/1.1",
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30000),
    });
  } catch (e) {
    return json({ error: "kaggle api unreachable", detail: String(e).slice(0, 200) }, 502);
  }
  let data = {};
  try { data = await resp.json(); } catch {}
  if (!resp.ok || (data && data.error)) {
    const detail = (data && data.error && (data.error.message || data.error)) || `kaggle http ${resp.status}`;
    return json({ error: "kaggle rejected the push", detail: String(detail).slice(0, 300) }, 502);
  }
  return json({
    ok: true,
    version: data.versionNumber ?? null,
    url: data.url ?? null,
    detail: "T4s booting — ComfyUI + LTX-2.5 weights fetch on boot, queue renders in ~10–15 min",
  });
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;
    if (!path.startsWith("/api/")) return env.ASSETS.fetch(request);
    try {
      if (path === "/api/knock" && request.method === "POST") {
        const state = await getState(env);
        const expected = state.panel_pin || "";
        let got = "";
        try { got = (await request.json()).pin || ""; } catch {}
        if (!expected) return json({ error: "not provisioned" }, 503);
        if (!timingSafeEqual(got, expected)) return json({ error: "bad PIN" }, 401);
        const resp = json({ ok: true });
        resp.headers.append("set-cookie", `${PIN_COOKIE}=${encodeURIComponent(got)}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000`);
        return resp;
      }

      if (path === "/api/jobs" && request.method === "GET") {
        const { state, err } = await requirePin(request, env, url);
        if (err) return err;
        const [statuses, boot] = await Promise.all([listStatuses(env), getBootMarker(env)]);
        const index = await getIndex(env, statuses.map((s) => s.id));
        const jobs = statuses.map((s) => ({
          ...s,
          prompt: index[s.id]?.prompt || "",
          mode: index[s.id]?.mode || null,
          aspect: index[s.id]?.aspect || null,
          has_audio: index[s.id]?.audio !== false,
          created_at: index[s.id]?.created_at || null,
          media: s.status === "DONE"
            ? { video: `/api/media/${PREFIX}/results/${s.id}/video.mp4?v=${MEDIA_VER}`, thumb: `/api/media/${PREFIX}/results/${s.id}/thumb.jpg?v=${MEDIA_VER}` }
            : null,
        }));
        const wake_age = state.wake_ts ? Math.round(Date.now() / 1e3) - state.wake_ts : null;
        const boot_ts = boot ? boot.at : null;
        const boot_age = boot_ts ? Date.now() / 1e3 - boot_ts : null;
        // Boot age alone lies (marker survives cancel/idle-exit). Honest liveness:
        // fresh boot (< 20 min — covers the ~15 min weights fetch with no heartbeats)
        // OR a kernel status heartbeat in the last 5 min.
        const last_activity = statuses.reduce((m, s) => Math.max(m, s.updated_at || 0), 0) || null;
        const last_activity_age = last_activity ? Date.now() / 1e3 - last_activity : null;
        const session_live = (boot_age !== null && boot_age < 20 * 60) || (last_activity_age !== null && last_activity_age < 300);
        return json({
          jobs,
          lane: {
            waking_recently: wake_age !== null && wake_age < WAKE_COOLDOWN_S,
            wake_ts: state.wake_ts || null,
            pending: jobs.filter((j) => j.status === "QUEUED").length,
            rendering: jobs.filter((j) => j.status === "RENDERING").length,
            boot_ts,
            session_age_min: boot_age !== null ? boot_age / 60 : null,
            last_activity_ts: last_activity,
            session_live,
            engine: ENGINE,
          },
        });
      }

      if (path === "/api/jobs" && request.method === "POST") {
        const { err } = await requirePin(request, env, url);
        if (err) return err;
        let body;
        try { body = await request.json(); } catch { return json({ error: "bad json" }, 400); }
        const mode = body.mode === "i2v" ? "i2v" : body.mode === "t2v" ? "t2v" : null;
        if (!mode) return json({ error: "mode must be t2v or i2v" }, 400);
        const prompt = String(body.prompt || "").trim();
        if (!prompt) return json({ error: "prompt is required" }, 400);
        if (prompt.length > 2000) return json({ error: "prompt too long (max 2000)" }, 400);
        const aspect = body.aspect === "landscape" ? "landscape" : "portrait";
        const duration_s = body.duration_s === undefined || body.duration_s === null || body.duration_s === "" ? 5 : Number(body.duration_s);
        if (![5, 15].includes(duration_s)) return json({ error: "duration_s must be 5 or 15" }, 400);
        let seed = null;
        if (body.seed !== undefined && body.seed !== null && String(body.seed) !== "") {
          const n = Number(body.seed);
          if (!Number.isInteger(n) || n < 0 || n > 4294967295) return json({ error: "seed must be an integer 0..4294967295" }, 400);
          seed = n;
        }
        const audio = body.audio === false ? false : true;
        const id = crypto.randomUUID().replace(/-/g, "").slice(0, 24);
        const spec = { id, mode, prompt, aspect, duration_s, seed, audio, super_res: false, created_at: new Date().toISOString() };
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
          prompt, mode, aspect, duration_s, audio, created_at: spec.created_at,
        }), { httpMetadata: { contentType: "application/json" } });
        return json({ ok: true, id, detail: "queued on the exchange" });
      }

      const mediaMatch = path.match(/^\/api\/media\/(.+)$/);
      if (mediaMatch && request.method === "GET") {
        const { err } = await requirePin(request, env, url);
        if (err) return err;
        const key = decodeURIComponent(mediaMatch[1]).replace(/\?.*$/, "");
        if (!key.startsWith(`${PREFIX}/results/`)) return json({ error: "not a media key" }, 400);
        if (!/\.(mp4|jpg|jpeg|png)$/.test(key)) return json({ error: "bad media type" }, 400);
        const obj = await env.CTRL.get(key);
        if (!obj) return json({ error: "not found" }, 404);
        const ct = key.endsWith(".mp4") ? "video/mp4" : key.endsWith(".png") ? "image/png" : "image/jpeg";
        return new Response(obj.body, {
          headers: { "content-type": ct, "cache-control": "private, max-age=86400" },
        });
      }

      if (path === "/api/wake" && request.method === "POST") {
        return wakeKernel(request, env, url);
      }

      return json({ error: "unknown api route" }, 404);
    } catch (e) {
      return json({ error: "worker exception", detail: String(e).slice(0, 200) }, 500);
    }
  },
};
