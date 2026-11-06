import { NextResponse } from "next/server";
import { getManifest } from "@/lib/memo";

export const dynamic = "force-dynamic";

export async function GET() {
  const m = await getManifest();
  if (!m) return NextResponse.json({ error: "manifest not generated yet" }, { status: 404 });
  return NextResponse.json({
    updated_at: m.updated_at,
    r2_mirrored: m.r2_mirrored,
    agent_todo: m.agent_todo_json,
    agent_status_md: m.agent_status_md,
  });
}
