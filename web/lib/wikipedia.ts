import { ask, type AskResult, type Evidence, type SystemId } from "@/lib/engine";

const API = "https://en.wikipedia.org/w/api.php";
const AGENT = "AdaptiveTemporalRAG/1.0 (https://adaptive-temporal-rag.vercel.app)";

type WikiPage = {
  pageid: number;
  title: string;
  extract: string;
  timestamp: string;
};

export async function askWikipedia(query: string, system: SystemId): Promise<AskResult> {
  const started = performance.now();
  if (system === "closed") {
    const closed = ask(query, "closed");
    return { ...closed, source: "wikipedia" };
  }

  const pages = await searchPages(query);
  const evidence = pages.map((page, index) => toEvidence(page, index, pages.length));
  return {
    query,
    system,
    source: "wikipedia",
    answer: answerFrom(query, pages),
    didRetrieve: evidence.length > 0,
    latencyMs: Math.max(1, Math.round(performance.now() - started)),
    backend: "extractive",
    analyzer: null,
    window: null,
    sufficiency: evidence.length > 0 ? { sufficient: true, score: 1 } : { sufficient: false, score: 0 },
    evidence,
  };
}

async function searchPages(query: string): Promise<WikiPage[]> {
  const search = new URL(API);
  search.searchParams.set("action", "query");
  search.searchParams.set("list", "search");
  search.searchParams.set("srsearch", query);
  search.searchParams.set("srlimit", "4");
  search.searchParams.set("srnamespace", "0");
  search.searchParams.set("format", "json");
  search.searchParams.set("utf8", "1");
  const found = await wikiGet(search);
  const hits = (found.query?.search ?? []) as Array<{ pageid: number; title: string; timestamp?: string }>;
  if (hits.length === 0) return [];

  const detail = new URL(API);
  detail.searchParams.set("action", "query");
  detail.searchParams.set("prop", "extracts");
  detail.searchParams.set("pageids", hits.map((hit) => String(hit.pageid)).join("|"));
  detail.searchParams.set("explaintext", "1");
  detail.searchParams.set("exintro", "1");
  detail.searchParams.set("format", "json");
  detail.searchParams.set("utf8", "1");
  const body = await wikiGet(detail);
  const pages = body.query?.pages ?? {};
  const order = new Map(hits.map((hit, index) => [hit.pageid, index]));
  return hits
    .map((hit) => {
      const page = pages[String(hit.pageid)];
      const extract = (page?.extract ?? "").replace(/\s+/g, " ").trim();
      return {
        pageid: hit.pageid,
        title: page?.title || hit.title,
        extract,
        timestamp: (hit.timestamp ?? "").slice(0, 10),
      };
    })
    .filter((page) => page.extract && !/may refer to:/i.test(page.extract))
    .sort((a, b) => (order.get(a.pageid) ?? 99) - (order.get(b.pageid) ?? 99));
}

async function wikiGet(url: URL): Promise<{ error?: { info?: string }; query?: { search?: unknown[]; pages?: Record<string, { title?: string; extract?: string }> } }> {
  const response = await fetch(url, {
    headers: { "User-Agent": AGENT, Accept: "application/json" },
    cache: "no-store",
    signal: AbortSignal.timeout(8000),
  });
  if (!response.ok) throw new Error(`Wikipedia returned ${response.status}`);
  const body = await response.json();
  if (body.error) throw new Error(body.error.info || "Wikipedia rejected the request.");
  return body;
}

function toEvidence(page: WikiPage, index: number, total: number): Evidence {
  return {
    docId: `wiki-${page.pageid}`,
    timestamp: page.timestamp,
    title: page.title,
    entity: "Wikipedia",
    text: page.extract.length > 700 ? `${page.extract.slice(0, 697).trim()}…` : page.extract,
    score: Number((1 - index / Math.max(total, 1)).toFixed(3)),
    url: `https://en.wikipedia.org/wiki/${encodeURIComponent(page.title.replace(/ /g, "_"))}`,
  };
}

function answerFrom(query: string, pages: WikiPage[]): string {
  if (pages.length === 0) return `No Wikipedia page matched “${query}”.`;
  const words = query
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .split(/\s+/)
    .filter((term) => term.length > 2 && !["who", "what", "the", "was", "were", "how", "did", "does"].includes(term));
  const topic = words.filter((term) => term !== "current" && term !== "currently");
  const whoNow = /\bwho(?:'s|s)?\b/i.test(query) && !/\b(was|were|in \d{4})\b/i.test(query);
  const primary = pages.reduce((best, page, index) => {
    const score = titleScore(page.title, topic);
    if (score > best.score || (score === best.score && index < best.index)) return { score, index };
    return best;
  }, { score: -1, index: 0 });
  const ranked = sentences(pages[primary.index].extract).map((sentence) => ({
    sentence,
      score: overlap(words, sentence) - (whoNow && /\b(?:(?:was|were) the|former|until|succeeded)\b/i.test(sentence) ? 3 : 0),
  }));
  ranked.sort((a, b) => b.score - a.score);
  const best = ranked[0]?.sentence || sentences(pages[0].extract)[0] || pages[0].extract;
  return best.length > 420 ? `${best.slice(0, 417).trim()}…` : best;
}

function sentences(extract: string): string[] {
  return extract
    .split(/(?<=[.!?])\s+/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length > 40);
}

function titleScore(title: string, terms: string[]): number {
  const words = title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .split(/\s+/)
    .filter((word) => word.length > 2 && !["the", "and", "for"].includes(word));
  if (words.length === 0 || terms.length === 0) return 0;
  const hits = words.filter((word) => terms.includes(word)).length;
  return hits / words.length;
}

function overlap(terms: string[], sentence: string): number {
  const text = sentence.toLowerCase();
  return terms.reduce((hits, term) => hits + (text.includes(term) ? 1 : 0), 0);
}
