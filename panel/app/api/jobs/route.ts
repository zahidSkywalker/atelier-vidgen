import { NextRequest, NextResponse } from "next/server";
import { db } from "@/lib/db";
import { promises as fs } from "fs";
import path from "path";
import crypto from "crypto";
import { MAX_PROMPT } from "@/lib/vidgen";

export const runtime = "nodejs";

const UPLOAD_DIR = path.join(process.cwd(), "public", "uploads");

// GET /api/jobs — list all jobs, newest first
export async function GET() {
  const jobs = await db.job.findMany({
    orderBy: { createdAt: "desc" },
    take: 120,
  });
  return NextResponse.json({ jobs });
}

// POST /api/jobs — submit a T2V or I2V render job (multipart/form-data)
export async function POST(req: NextRequest) {
  try {
    const form = await req.formData();
    const mode = String(form.get("mode") || "t2v");
    if (mode !== "t2v" && mode !== "i2v") {
      return NextResponse.json({ error: "mode must be t2v or i2v" }, { status: 400 });
    }
    const prompt = String(form.get("prompt") || "").trim();
    if (!prompt) {
      return NextResponse.json({ error: "prompt is required" }, { status: 400 });
    }
    if (prompt.length > MAX_PROMPT) {
      return NextResponse.json({ error: `prompt too long (max ${MAX_PROMPT})` }, { status: 400 });
    }
    const aspect = String(form.get("aspect") || "portrait");
    if (aspect !== "portrait" && aspect !== "landscape") {
      return NextResponse.json({ error: "aspect must be portrait or landscape" }, { status: 400 });
    }
    const audio = String(form.get("audio") || "true") === "true";
    const superRes = String(form.get("superRes") || "false") === "true";

    let seed: number | null = null;
    const rawSeed = String(form.get("seed") || "").trim();
    if (rawSeed) {
      const n = Number(rawSeed);
      if (!Number.isInteger(n) || n < 0 || n > 4294967295) {
        return NextResponse.json({ error: "seed must be an integer 0..4294967295" }, { status: 400 });
      }
      seed = n;
    }

    let refImage: string | null = null;
    if (mode === "i2v") {
      const file = form.get("ref");
      if (!(file instanceof File) || file.size === 0) {
        return NextResponse.json({ error: "i2v requires a reference image" }, { status: 400 });
      }
      if (file.size > 12 * 1024 * 1024) {
        return NextResponse.json({ error: "reference image too large (max 12 MB)" }, { status: 400 });
      }
      const buf = Buffer.from(await file.arrayBuffer());
      const kind = file.type === "image/png" ? "png" : "jpg";
      await fs.mkdir(UPLOAD_DIR, { recursive: true });
      const id = crypto.randomBytes(8).toString("hex");
      const name = `ref_${id}.${kind}`;
      await fs.writeFile(path.join(UPLOAD_DIR, name), buf);
      refImage = `/uploads/${name}`;
    }

    const job = await db.job.create({
      data: { mode, prompt, aspect, seed, audio, superRes, refImage, status: "QUEUED" },
    });

    return NextResponse.json({ job }, { status: 201 });
  } catch (err) {
    console.error("POST /api/jobs failed:", err);
    return NextResponse.json({ error: "submit failed" }, { status: 500 });
  }
}
