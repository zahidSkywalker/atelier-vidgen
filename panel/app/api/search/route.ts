import { NextRequest, NextResponse } from "next/server";
import { semanticAuditSearch } from "@/lib/memo";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const q = (req.nextUrl.searchParams.get("q") || "").trim();
  const limit = Math.min(20, Math.max(1, parseInt(req.nextUrl.searchParams.get("limit") || "5", 10)));
  if (!q) return NextResponse.json({ error: "q required" }, { status: 400 });
  try {
    const hits = await semanticAuditSearch(q, limit);
    return NextResponse.json({
      query: q,
      hits: hits.map((h) => ({
        score: h.score,
        task_id: h.task_id,
        phase: h.phase,
        timestamp: h.timestamp,
        log: h.log,
      })),
    });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
