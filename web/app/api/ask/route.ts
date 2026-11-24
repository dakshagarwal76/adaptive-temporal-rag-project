import { NextResponse } from "next/server";
import { ask, type SystemId } from "@/lib/engine";

const SYSTEMS = new Set<SystemId>(["adaptive", "standard", "fixed", "closed"]);

export async function POST(request: Request) {
  let body: { query?: unknown; system?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Send a question as JSON." }, { status: 400 });
  }
  const query = typeof body.query === "string" ? body.query.trim().slice(0, 400) : "";
  const system = typeof body.system === "string" && SYSTEMS.has(body.system as SystemId)
    ? (body.system as SystemId)
    : "adaptive";
  if (!query) {
    return NextResponse.json({ error: "Enter a question." }, { status: 400 });
  }
  return NextResponse.json(ask(query, system));
}
