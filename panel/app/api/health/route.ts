import { NextResponse } from "next/server";
import { cols } from "@/lib/memo";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const { tasks, logs, vectors } = await cols();
    const [t, l, v] = await Promise.all([tasks.countDocuments(), logs.countDocuments(), vectors.countDocuments()]);
    return NextResponse.json({ ok: true, service: "MemO", tasks: t, logs: l, vectors: v, ts: new Date().toISOString() });
  } catch (e) {
    return NextResponse.json({ ok: false, error: String(e) }, { status: 500 });
  }
}
