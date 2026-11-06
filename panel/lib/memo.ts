import crypto from "crypto";
import { MongoClient, Collection, Db } from "mongodb";

/* ------------------------------------------------------------------ */
/* Mongo client singleton (reused across serverless invocations)      */
/* ------------------------------------------------------------------ */

const g = globalThis as unknown as { __memoClient?: MongoClient };

export const DIM = 384;

export interface TaskDoc {
  _id: string; // TASK-YYYYMMDD-NN
  title: string;
  description: string;
  status: "PENDING" | "IN_PROGRESS" | "COMPLETED" | "FAILED";
  created_at: Date;
  updated_at: Date;
}

export interface LogDoc {
  _id: string;
  task_id: string;
  timestamp: Date;
  phase: "PLANNING" | "TOOL_CALL" | "VERIFICATION" | "SYSTEM_HANDOFF";
  agent_reasoning: string;
  action_payload: string;
  execution_result: string;
}

export interface VectorDoc {
  _id: string;
  log_id: string;
  task_id: string;
  phase: string;
  timestamp: Date;
  text: string;
  embedding: number[];
  embed_source: "workers-ai" | "huggingface" | "local-hash";
}

export const STATUSES = ["PENDING", "IN_PROGRESS", "COMPLETED", "FAILED"] as const;
export const PHASES = ["PLANNING", "TOOL_CALL", "VERIFICATION", "SYSTEM_HANDOFF"] as const;

export async function memoDb(): Promise<Db> {
  if (!g.__memoClient) {
    const uri = process.env.MONGODB_URI;
    if (!uri) throw new Error("MONGODB_URI not configured");
    const cl = new MongoClient(uri, { serverSelectionTimeoutMS: 15000 });
    await cl.connect();
    g.__memoClient = cl;
  }
  return g.__memoClient.db("memo");
}

export async function cols() {
  const db = await memoDb();
  return {
    tasks: db.collection<TaskDoc>("tasks"),
    logs: db.collection<LogDoc>("execution_logs"),
    vectors: db.collection<VectorDoc>("log_vectors"),
    counters: db.collection("counters"),
    manifest: db.collection<ManifestDoc>("manifest"),
  };
}

/* ------------------------------------------------------------------ */
/* Embeddings — HuggingFace bge-small (384d), local hash fallback      */
/* ------------------------------------------------------------------ */

function l2(v: number[]): number[] {
  const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
  return v.map((x) => x / n);
}

function hashEmbed(text: string): number[] {
  const v = new Array(DIM).fill(0);
  const words = text.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/).filter(Boolean);
  for (const w of words) {
    let h = 2166136261;
    for (let i = 0; i < w.length; i++) {
      h ^= w.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    const sign = (h >>> 31) & 1 ? -1 : 1;
    v[Math.abs(h) % DIM] += sign;
  }
  return l2(v);
}

function pool(j: unknown): number[] | null {
  if (Array.isArray(j) && typeof j[0] === "number") return j as number[];
  if (Array.isArray(j) && Array.isArray(j[0])) {
    const toks = j as number[][];
    const out = new Array(toks[0].length).fill(0);
    for (const t of toks) for (let i = 0; i < out.length; i++) out[i] += t[i];
    return out.map((x) => x / (toks.length || 1));
  }
  if (j && typeof j === "object") {
    const emb = (j as Record<string, unknown>)["embeddings"];
    if (emb) return pool(emb);
  }
  return null;
}

export async function embed(text: string): Promise<{ vector: number[]; source: "workers-ai" | "huggingface" | "local-hash" }> {
  const input = text.slice(0, 3000);

  // primary: edge-platform AI inference (bge-small-en-v1.5, 384-d, token already vaulted)
  const acc = process.env.CF_ACCOUNT_ID;
  const cfTok = process.env.CF_API_TOKEN;
  if (acc && cfTok) {
    try {
      const r = await fetch(`https://api.cloudflare.com/client/v4/accounts/${acc}/ai/run/@cf/baai/bge-small-en-v1.5`, {
        method: "POST",
        headers: { authorization: `Bearer ${cfTok}`, "content-type": "application/json" },
        body: JSON.stringify({ text: [input] }),
        signal: AbortSignal.timeout(20000),
      });
      if (r.ok) {
        const j = await r.json();
        const v = j?.result?.data?.[0];
        if (Array.isArray(v) && v.length === DIM) return { vector: v, source: "workers-ai" };
      }
    } catch {
      /* fall through */
    }
  }

  // secondary: hosted inference provider (may be credit-limited)
  const hf = process.env.HF_TOKEN;
  if (hf) {
    try {
      const r = await fetch(
        "https://router.huggingface.co/hf-inference/models/BAAI/bge-small-en-v1.5/pipeline/feature-extraction",
        {
          method: "POST",
          headers: { authorization: `Bearer ${hf}`, "content-type": "application/json" },
          body: JSON.stringify({ inputs: input }),
          signal: AbortSignal.timeout(20000),
        }
      );
      if (r.ok) {
        const v = pool(await r.json());
        if (v && v.length === DIM) return { vector: l2(v), source: "huggingface" };
      }
    } catch {
      /* fall through to local hash */
    }
  }
  return { vector: hashEmbed(input), source: "local-hash" };
}

/* ------------------------------------------------------------------ */
/* Core services — task lifecycle + append-only ledger + vectorization */
/* ------------------------------------------------------------------ */

function dayStamp(d = new Date()): string {
  return `${d.getUTCFullYear()}${String(d.getUTCMonth() + 1).padStart(2, "0")}${String(d.getUTCDate()).padStart(2, "0")}`;
}

export async function createTask(title: string, description = ""): Promise<TaskDoc> {
  const { tasks, counters } = await cols();
  const counterId = `task_seq_${dayStamp()}`;
  const r = await counters.findOneAndUpdate(
    { counter_id: counterId },
    { $inc: { seq: 1 } },
    { upsert: true, returnDocument: "after" }
  );
  const seq = (r?.seq ?? 1) as number;
  const id = `TASK-${dayStamp()}-${String(seq).padStart(2, "0")}`;
  const now = new Date();
  const doc: TaskDoc = {
    _id: id,
    title: title.slice(0, 300),
    description: (description || "").slice(0, 4000),
    status: "PENDING",
    created_at: now,
    updated_at: now,
  };
  await tasks.insertOne(doc);
  await regenerateManifest();
  return doc;
}

export async function updateTaskStatus(taskId: string, status: TaskDoc["status"]): Promise<TaskDoc | null> {
  const { tasks } = await cols();
  if (!STATUSES.includes(status)) throw new Error(`invalid status: ${status}`);
  const doc = await tasks.findOneAndUpdate(
    { _id: taskId },
    { $set: { status, updated_at: new Date() } },
    { returnDocument: "after" }
  );
  if (doc) await regenerateManifest();
  return doc;
}

export interface LogStepInput {
  task_id: string;
  phase: LogDoc["phase"];
  agent_reasoning: string;
  action_payload: string;
  execution_result: string;
}

/**
 * Append an immutable ledger entry. There is NO update/delete path for
 * execution_logs anywhere in this codebase — history cannot be rewritten.
 * On commit, the audit string is embedded and upserted into the vector store.
 */
export async function logStep(input: LogStepInput): Promise<{ log: LogDoc; vector_id: string; embed_source: string }> {
  const { tasks, logs, vectors } = await cols();
  if (!PHASES.includes(input.phase)) throw new Error(`invalid phase: ${input.phase}`);
  const task = await tasks.findOne({ _id: input.task_id });
  if (!task) throw new Error(`unknown task_id: ${input.task_id}`);

  const ts = new Date();
  const log: LogDoc = {
    _id: crypto.randomUUID(),
    task_id: input.task_id,
    timestamp: ts,
    phase: input.phase,
    agent_reasoning: (input.agent_reasoning || "").slice(0, 20000),
    action_payload: (input.action_payload || "").slice(0, 20000),
    execution_result: (input.execution_result || "").slice(0, 60000),
  };
  // append-only commit — insert only, never update, never delete
  await logs.insertOne(log);
  await tasks.updateOne({ _id: input.task_id }, { $set: { updated_at: ts } });

  const auditString = `task ${log.task_id} [${log.phase}] action: ${log.action_payload} result: ${log.execution_result} reasoning: ${log.agent_reasoning}`;
  const { vector, source } = await embed(auditString);
  const vectorDoc: VectorDoc = {
    _id: crypto.randomUUID(),
    log_id: log._id,
    task_id: log.task_id,
    phase: log.phase,
    timestamp: ts,
    text: auditString.slice(0, 4000),
    embedding: vector,
    embed_source: source,
  };
  await vectors.insertOne(vectorDoc); // upsert-by-uuid: idempotent insert, never overwritten
  await regenerateManifest();
  return { log, vector_id: vectorDoc._id, embed_source: source };
}

export interface AuditHit {
  log: LogDoc | null;
  task_id: string;
  phase: string;
  timestamp: Date;
  score: number;
}

export async function semanticAuditSearch(query: string, limit = 5): Promise<AuditHit[]> {
  const { vectors, logs } = await cols();
  const { vector } = await embed(query);
  const pipeline = [
    {
      $vectorSearch: {
        index: "memo_vec_idx",
        path: "embedding",
        queryVector: vector,
        numCandidates: Math.max(200, limit * 40),
        limit,
      },
    },
    { $set: { score: { $meta: "vectorSearchScore" } } },
    {
      $lookup: {
        from: "execution_logs",
        localField: "log_id",
        foreignField: "_id",
        as: "log",
      },
    },
    { $project: { score: 1, task_id: 1, phase: 1, timestamp: 1, log: { $ifNull: [{ $arrayElemAt: ["$log", 0] }, null] } } },
  ];
  const hits = await vectors.aggregate<AuditHit>(pipeline).toArray();
  return hits;
}

/* ------------------------------------------------------------------ */
/* Manifest synchronization — agent_todo.json + .agent_status.md       */
/* ------------------------------------------------------------------ */

export interface ManifestDoc {
  _id: string;
  updated_at: Date;
  agent_todo_json: unknown;
  agent_status_md: string;
  r2_mirrored: boolean;
}

export async function buildManifest(): Promise<Omit<ManifestDoc, "_id">> {
  const { tasks, logs } = await cols();
  const all = await tasks.find().sort("created_at", 1).toArray();
  const counts = { total: all.length } as Record<string, number>;
  for (const s of STATUSES) counts[s] = all.filter((t) => t.status === s).length;
  const lastLogs = await logs.find().sort("timestamp", -1).limit(5).toArray();

  const todo = {
    generated_at: new Date().toISOString(),
    instruction: "Read this file at the start of every run — it holds your current boundaries and remaining steps.",
    counts,
    tasks: all.map((t) => ({
      id: t._id,
      title: t.title,
      status: t.status,
      remaining: t.status === "COMPLETED" || t.status === "FAILED" ? "closed" : "active — continue or close",
      updated_at: t.updated_at,
    })),
    latest_activity: lastLogs.map((l) => ({
      at: l.timestamp,
      task: l.task_id,
      phase: l.phase,
      action: l.action_payload.slice(0, 140),
    })),
  };

  const md = [
    "# MemO — Agent Status",
    "",
    `_generated: ${todo.generated_at}_`,
    "",
    `**Tasks:** ${counts.total} total · ${counts.PENDING ?? 0} pending · ${counts.IN_PROGRESS ?? 0} in progress · ${counts.COMPLETED ?? 0} completed · ${counts.FAILED ?? 0} failed`,
    "",
    "## Tasks",
    ...all.map((t) => `- [${t.status}] **${t._id}** ${t.title}`),
    "",
    "## Latest ledger activity",
    ...lastLogs.map((l) => `- \`${l.timestamp.toISOString()}\` ${l.task_id} · ${l.phase} — ${l.action_payload.slice(0, 120)}`),
    "",
  ].join("\n");

  return { updated_at: new Date(), agent_todo_json: todo, agent_status_md: md, r2_mirrored: false };
}

export async function regenerateManifest(): Promise<ManifestDoc> {
  const { manifest } = await cols();
  const body = await buildManifest();
  const doc: ManifestDoc = { _id: "latest", ...body };
  await manifest.replaceOne({ _id: "latest" }, doc, { upsert: true });
  // mirror to object storage so any machine can read the manifest cheaply
  const okJson = await r2Put("memo/agent_todo.json", JSON.stringify(body.agent_todo_json, null, 2));
  const okMd = await r2Put("memo/agent_status.md", body.agent_status_md);
  if (okJson && okMd) {
    doc.r2_mirrored = true;
    await manifest.replaceOne({ _id: "latest" }, doc, { upsert: true });
  }
  return doc;
}

export async function getManifest(): Promise<ManifestDoc | null> {
  const { manifest } = await cols();
  return manifest.findOne({ _id: "latest" });
}

/* ------------------------------------------------------------------ */
/* Minimal S3 SigV4 PUT (object-storage mirror, non-blocking)          */
/* ------------------------------------------------------------------ */

async function r2Put(key: string, body: string): Promise<boolean> {
  const ep = process.env.R2_ENDPOINT;
  const acc = process.env.R2_ACCESS_KEY;
  const sec = process.env.R2_SECRET_KEY;
  const bucket = process.env.R2_BUCKET;
  if (!ep || !acc || !sec || !bucket) return false;
  try {
    const host = new URL(ep).host;
    const region = "auto";
    const amzDate = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}/, "");
    const dateStamp = amzDate.slice(0, 8);
    const payloadHash = crypto.createHash("sha256").update(body).digest("hex");
    const canonicalRequest = [
      "PUT",
      `/${bucket}/${key}`,
      "",
      `host:${host}`,
      `x-amz-content-sha256:${payloadHash}`,
      `x-amz-date:${amzDate}`,
      "",
      "host;x-amz-content-sha256;x-amz-date",
      payloadHash,
    ].join("\n");
    const scope = `${dateStamp}/${region}/s3/aws4_request`;
    const stringToSign = [
      "AWS4-HMAC-SHA256",
      amzDate,
      scope,
      crypto.createHash("sha256").update(canonicalRequest).digest("hex"),
    ].join("\n");
    const hmac = (k: crypto.BinaryLike | crypto.KeyObject, d: string) => crypto.createHmac("sha256", k).update(d).digest();
    const kSigning = hmac(hmac(hmac(hmac(`AWS4${sec}`, dateStamp), region), "s3"), "aws4_request");
    const signature = crypto.createHmac("sha256", kSigning).update(stringToSign).digest("hex");
    const r = await fetch(`${new URL(ep).origin}/${bucket}/${key}`, {
      method: "PUT",
      headers: {
        Authorization: `AWS4-HMAC-SHA256 Credential=${acc}/${scope}, SignedHeaders=host;x-amz-content-sha256;x-amz-date, Signature=${signature}`,
        "x-amz-date": amzDate,
        "x-amz-content-sha256": payloadHash,
      },
      body,
      signal: AbortSignal.timeout(15000),
    });
    return r.ok;
  } catch {
    return false;
  }
}

/* ------------------------------------------------------------------ */
/* Write auth                                                          */
/* ------------------------------------------------------------------ */

export function authorized(req: Request): boolean {
  const key = process.env.MEMO_WRITE_KEY;
  if (!key) return false; // fails closed
  return req.headers.get("x-memo-key") === key;
}
