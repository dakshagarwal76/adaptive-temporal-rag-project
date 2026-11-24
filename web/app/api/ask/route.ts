import { NextResponse } from "next/server";
import { ask, type SourceId, type SystemId } from "@/lib/engine";
import { askWikipedia } from "@/lib/wikipedia";

const SYSTEMS = new Set<SystemId>(["adaptive", "standard", "fixed", "closed"]);
const SOURCES = new Set<SourceId>(["corpus", "wikipedia"]);

export async function POST(request: Request) {
  let body: { query?: unknown; system?: unknown; source?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Send a question as JSON." }, { status: 400 });
  }
  const query = typeof body.query === "string" ? body.query.trim().slice(0, 400) : "";
  const system = typeof body.system === "string" && SYSTEMS.has(body.system as SystemId)
    ? (body.system as SystemId)
    : "adaptive";
  const source = typeof body.source === "string" && SOURCES.has(body.source as SourceId)
    ? (body.source as SourceId)
    : "corpus";
  if (!query) {
    return NextResponse.json({ error: "Enter a question." }, { status: 400 });
  }
  if (source === "wikipedia") {
    try {
      return NextResponse.json(await askWikipedia(query, system));
    } catch {
      return NextResponse.json({ error: "Wikipedia could not be reached." }, { status: 502 });
    }
  }
  return NextResponse.json(ask(query, system));
}
