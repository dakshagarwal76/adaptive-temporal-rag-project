import { ask, type AskResult, type Evidence, type SystemId } from "@/lib/engine";

const API = "https://en.wikipedia.org/w/api.php";
const AGENT = "AdaptiveTemporalRAG/1.0 (https://adaptive-temporal-rag.vercel.app)";
const STOP = new Set(["who", "what", "which", "when", "where", "whom", "whose", "whos", "the", "a", "an", "of", "is", "are", "was", "were", "did", "does", "how", "for", "and"]);

type WikiPage = {
  pageid: number;
  title: string;
  extract: string;
  timestamp: string;
  score: number;
};

export async function askWikipedia(query: string, system: SystemId): Promise<AskResult> {
  const started = performance.now();
  if (system === "closed") {
    const closed = ask(query, "closed");
    return { ...closed, source: "wikipedia" };
  }

  const pages = await searchPages(query);
  const terms = queryTerms(query);
  const ranked = pages
    .map((page) => ({ ...page, score: relevance(terms, page.title, page.extract) }))
    .sort((a, b) => b.score - a.score || titlePrecision(terms, b.title) - titlePrecision(terms, a.title));
  return {
    query,
    system,
    source: "wikipedia",
    answer: answerFrom(query, terms, ranked),
    didRetrieve: ranked.length > 0,
    latencyMs: Math.max(1, Math.round(performance.now() - started)),
    backend: "extractive",
    analyzer: null,
    window: null,
    sufficiency: ranked.length > 0 ? { sufficient: ranked[0].score >= 0.34, score: ranked[0]?.score ?? 0 } : { sufficient: false, score: 0 },
    evidence: ranked.map(toEvidence),
  };
}

async function searchPages(query: string): Promise<WikiPage[]> {
  const search = new URL(API);
  search.searchParams.set("action", "query");
  search.searchParams.set("list", "search");
  search.searchParams.set("srsearch", query);
  search.searchParams.set("srlimit", "50");
  search.searchParams.set("srnamespace", "0");
  search.searchParams.set("format", "json");
  search.searchParams.set("utf8", "1");
  const found = await wikiGet(search);
  const hits = (found.query?.search ?? []) as Array<{ pageid: number; title: string; timestamp?: string }>;
  if (hits.length === 0) return [];

  const pages: WikiPage[] = [];
  for (let start = 0; start < hits.length; start += 20) {
    const batch = hits.slice(start, start + 20);
    const detail = new URL(API);
    detail.searchParams.set("action", "query");
    detail.searchParams.set("prop", "extracts");
    detail.searchParams.set("pageids", batch.map((hit) => String(hit.pageid)).join("|"));
    detail.searchParams.set("explaintext", "1");
    detail.searchParams.set("exintro", "1");
    detail.searchParams.set("format", "json");
    detail.searchParams.set("utf8", "1");
    const body = await wikiGet(detail);
    const byId = body.query?.pages ?? {};
    for (const hit of batch) {
      const page = byId[String(hit.pageid)];
      const extract = (page?.extract ?? "").replace(/\s+/g, " ").trim();
      if (!extract || /may refer to:/i.test(extract)) continue;
      pages.push({
        pageid: hit.pageid,
        title: page?.title || hit.title,
        extract,
        timestamp: (hit.timestamp ?? "").slice(0, 10),
        score: 0,
      });
    }
  }
  return pages;
}

async function wikiGet(url: URL): Promise<{ error?: { info?: string }; query?: { search?: unknown[]; pages?: Record<string, { title?: string; extract?: string }> } }> {
  const response = await fetch(url, {
    headers: { "User-Agent": AGENT, Accept: "application/json" },
    cache: "no-store",
    signal: AbortSignal.timeout(20000),
  });
  if (!response.ok) throw new Error(`Wikipedia returned ${response.status}`);
  const body = await response.json();
  if (body.error) throw new Error(body.error.info || "Wikipedia rejected the request.");
  return body;
}

function toEvidence(page: WikiPage): Evidence {
  return {
    docId: `wiki-${page.pageid}`,
    timestamp: page.timestamp,
    title: page.title,
    entity: "Wikipedia",
    text: page.extract,
    score: page.score,
    url: `https://en.wikipedia.org/wiki/${encodeURIComponent(page.title.replace(/ /g, "_"))}`,
  };
}

function answerFrom(query: string, terms: string[], pages: WikiPage[]): string {
  if (pages.length === 0) return `No Wikipedia page matched “${query}”.`;
  const page = pages[0];
  const parts = sentences(page.extract).filter((sentence) => overlap(terms, sentence) > 0);
  const chosen = parts.length > 0 ? parts : sentences(page.extract).slice(0, 2);
  return chosen.join(" ") || page.extract;
}

function sentences(extract: string): string[] {
  return extract
    .split(/(?<!\b[A-Z])(?<=[.!?])\s+(?=[A-Z])/)
    .map((sentence) => sentence.trim())
    .filter((sentence) => sentence.length > 40);
}

function queryTerms(query: string): string[] {
  return [...new Set(query.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/).filter((term) => term.length > 2 && !STOP.has(term)))];
}

function titlePrecision(terms: string[], title: string): number {
  const words = title.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/).filter((word) => word.length > 2 && !STOP.has(word));
  if (words.length === 0) return 0;
  const hits = words.filter((word) => terms.some((term) => word.includes(term))).length;
  return hits / words.length;
}

function relevance(terms: string[], title: string, text: string): number {
  if (terms.length === 0) return 0;
  const blob = `${title} ${text}`.toLowerCase();
  const titleText = title.toLowerCase();
  const hits = terms.filter((term) => blob.includes(term)).length;
  const titleHits = terms.filter((term) => titleText.includes(term)).length;
  return Number(((0.7 * hits + 0.3 * titleHits) / terms.length).toFixed(3));
}

function overlap(terms: string[], sentence: string): number {
  const text = sentence.toLowerCase();
  return terms.reduce((hits, term) => hits + (text.includes(term) ? 1 : 0), 0);
}
