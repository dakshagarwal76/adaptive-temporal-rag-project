import { NextResponse } from "next/server";
import { ask, type AskResult, type SourceId, type SystemId } from "@/lib/engine";
import { fallbackAnswer, generateAnswer } from "@/lib/generate";
import { askGdelt } from "@/lib/gdelt";
import { askWikidata } from "@/lib/wikidata";
import { askWikipedia } from "@/lib/wikipedia";

const SYSTEMS = new Set<SystemId>(["adaptive", "standard", "fixed", "closed"]);
const ORDER: SourceId[] = ["corpus", "wikipedia", "wikidata", "gdelt"];
const SOURCE_SET = new Set<SourceId>(ORDER);

export const maxDuration = 30;

export async function POST(request: Request) {
  let body: { query?: unknown; system?: unknown; source?: unknown; sources?: unknown };
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Send a question as JSON." }, { status: 400 });
  }
  const query = typeof body.query === "string" ? body.query.trim().slice(0, 400) : "";
  const system = typeof body.system === "string" && SYSTEMS.has(body.system as SystemId)
    ? (body.system as SystemId)
    : "adaptive";
  const sources = parseSources(body);
  if (!query) {
    return NextResponse.json({ error: "Enter a question." }, { status: 400 });
  }
  const started = performance.now();
  if (system === "closed") {
    const closed = finish(ask(query, "closed"), sources, 0);
    return NextResponse.json(await timed(closed, started));
  }
  const settled = await Promise.all(sources.map(async (source) => {
    try {
      return { ok: true as const, source, result: await askOne(query, system, source) };
    } catch (error) {
      const message = error instanceof Error ? error.message : "That source could not be reached.";
      return { ok: false as const, source, message };
    }
  }));
  const ready = settled.flatMap((item) => (item.ok ? [item.result] : []));
  const failed = settled.flatMap((item) => (item.ok ? [] : [{ source: item.source, message: item.message }]));
  if (ready.length === 0) {
    return NextResponse.json({ error: failed.map((item) => item.message).join(" ") }, { status: 502 });
  }
  const combined = combine(ready, failed, sources, 1);
  return NextResponse.json(await timed(combined, started));
}

async function timed(result: AskResult, started: number): Promise<AskResult> {
  const answered = await withModel(result);
  return { ...answered, latencyMs: Math.max(1, Math.round(performance.now() - started)) };
}

async function withModel(result: AskResult): Promise<AskResult> {
  const generated = await generateAnswer(result.query, result.evidence);
  const pooled = fallbackAnswer(result.query, result.evidence);
  const warnings = [...(result.warnings ?? [])];
  if (generated?.error) warnings.push(generated.error);
  return {
    ...result,
    answer: generated?.answer || pooled || result.answer,
    backend: generated?.answer ? generated.backend : result.backend,
    warnings: warnings.length > 0 ? warnings : undefined,
  };
}

function parseSources(body: { source?: unknown; sources?: unknown }): SourceId[] {
  const raw = Array.isArray(body.sources) ? body.sources : typeof body.source === "string" ? [body.source] : ["corpus"];
  const picked = new Set(raw.filter((item): item is SourceId => typeof item === "string" && SOURCE_SET.has(item as SourceId)));
  const sources = ORDER.filter((source) => picked.has(source));
  return sources.length > 0 ? sources : ["corpus"];
}

async function askOne(query: string, system: SystemId, source: SourceId): Promise<AskResult> {
  if (source === "wikipedia") return askWikipedia(query, system);
  if (source === "wikidata") return askWikidata(query, system);
  if (source === "gdelt") return askGdelt(query, system);
  return ask(query, system);
}

function finish(result: AskResult, sources: SourceId[], latencyMs: number): AskResult {
  return {
    ...result,
    sources,
    latencyMs: latencyMs || result.latencyMs,
    evidence: result.evidence.map((item) => ({ ...item, origin: result.source })),
  };
}

function combine(results: AskResult[], failed: Array<{ source: SourceId; message: string }>, sources: SourceId[], latencyMs: number): AskResult {
  const bySource = new Map(results.map((result) => [result.source, finish(result, sources, latencyMs)]));
  const ordered = sources.flatMap((source) => {
    const result = bySource.get(source);
    return result ? [result] : [];
  });
  if (ordered.length === 1 && failed.length === 0) return ordered[0];
  const corpus = ordered.find((result) => result.source === "corpus");
  return {
    query: ordered[0].query,
    system: ordered[0].system,
    source: ordered[0].source,
    sources,
    warnings: failed.map((item) => item.message),
    answer: "",
    didRetrieve: ordered.some((result) => result.didRetrieve),
    latencyMs,
    backend: "extractive",
    analyzer: corpus?.analyzer ?? null,
    window: corpus?.window ?? null,
    sufficiency: corpus?.sufficiency ?? ordered.find((result) => result.sufficiency)?.sufficiency ?? null,
    evidence: ordered.flatMap((result) => result.evidence),
  };
}
