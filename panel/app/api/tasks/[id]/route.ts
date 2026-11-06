import { NextRequest, NextResponse } from "next/server";
import { cols, updateTaskStatus, STATUSES, authorized, TaskDoc } from "@/lib/memo";

export const dynamic = "force-dynamic";

export async function GET(_req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const { tasks, logs } = await cols();
  const task = await tasks.findOne({ _id: id });
  if (!task) return NextResponse.json({ error: "not found" }, { status: 404 });
  const timeline = await logs.find({ task_id: id }).sort("timestamp", 1).toArray();
  return NextResponse.json({ task, logs: timeline });
}

export async function PATCH(req: NextRequest, ctx: { params: Promise<{ id: string }> }) {
  if (!authorized(req)) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  const { id } = await ctx.params;
  try {
    const body = await req.json();
    const status = String(body.status || "") as TaskDoc["status"];
    if (!STATUSES.includes(status)) return NextResponse.json({ error: `status must be one of ${STATUSES.join("|")}` }, { status: 400 });
    const task = await updateTaskStatus(id, status);
    if (!task) return NextResponse.json({ error: "not found" }, { status: 404 });
    return NextResponse.json({ task });
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
