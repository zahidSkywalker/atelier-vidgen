import { NextRequest, NextResponse } from "next/server";
import { cols, logStep, authorized, PHASES, LogStepInput, LogDoc } from "@/lib/memo";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const taskId = req.nextUrl.searchParams.get("task_id");
  const { logs } = await cols();
  const q = taskId ? { task_id: taskId } : {};
  const timeline = await logs.find(q).sort("timestamp", -1).limit(500).toArray();
  return NextResponse.json({ logs: timeline, append_only: true });
}

export async function POST(req: NextRequest) {
  if (!authorized(req)) return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  try {
    const body = await req.json();
    const input: LogStepInput = {
      task_id: String(body.task_id || ""),
      phase: String(body.phase || "") as LogDoc["phase"],
      agent_reasoning: String(body.agent_reasoning ?? ""),
      action_payload: String(body.action_payload ?? ""),
      execution_result: String(body.execution_result ?? ""),
    };
    if (!input.task_id) return NextResponse.json({ error: "task_id required" }, { status: 400 });
    if (!PHASES.includes(input.phase)) return NextResponse.json({ error: `phase must be one of ${PHASES.join("|")}` }, { status: 400 });
    const out = await logStep(input);
    return NextResponse.json(
      {
        log_id: out.log._id,
        timestamp: out.log.timestamp,
        vector_id: out.vector_id,
        embed_source: out.embed_source,
        immutable: true,
      },
      { status: 201 }
    );
  } catch (e) {
    return NextResponse.json({ error: String(e) }, { status: 500 });
  }
}
