import { NextRequest, NextResponse } from "next/server";
import { cols, createTask, authorized } from "@/lib/memo";

export const dynamic = "force-dynamic";

export async function GET() {
  const { tasks } = await cols();
  const all = await tasks.find().sort("created_at", -1).toArray();
  return NextResponse.json({ tasks: all });
}

export async function POST(req: NextRequest) {
  if (!authorized(req)) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  try {
    const body = await req.json();
    const title = String(body.title || "").trim();
    if (!title) return NextResponse.json({ error: "title required" }, { status: 400 });
    const task = await createTask(title, String(body.description || ""));
    return NextResponse.json({ task }, { status: 201 });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
