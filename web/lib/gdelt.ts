import https from "node:https";
import { ask, type AskResult, type Evidence, type SystemId } from "@/lib/engine";

const API = "https://api.gdeltproject.org/api/v2/doc/doc";
const AGENT = "AdaptiveTemporalRAG/1.0 (https://adaptive-temporal-rag.vercel.app)";
const STOP = new Set(["who", "what", "which", "when", "where", "whom", "whose", "whos", "the", "a", "an", "of", "is", "are", "was", "were", "did", "does", "how", "for", "and", "current", "currently"]);

type Article = {
  url: string;
  title: string;
  seendate?: string;
  domain?: string;
  language?: string;
  sourcecountry?: string;
  score?: number;
};

export async function askGdelt(query: string, system: SystemId): Promise<AskResult> {
  const started = performance.now();
  if (system === "closed") {
    const closed = ask(query, "closed");
    return { ...closed, source: "gdelt" };
  }

  const articles = await searchArticles(query);
  const terms = keywords(query).toLowerCase().split(/\s+/).filter(Boolean);
  const ranked = articles
    .map((article) => ({ ...article, score: relevance(terms, article.title) }))
    .sort((a, b) => b.score - a.score || seenDay(b.seendate ?? "").localeCompare(seenDay(a.seendate ?? "")));
  const best = ranked[0];
  const lead = best ? await articleLead(best.url) : "";
  return {
    query,
    system,
    source: "gdelt",
    answer: best ? pointed(best.title, lead) : `No GDELT article matched “${query}” in the last 3 months.`,
    didRetrieve: ranked.length > 0,
    latencyMs: Math.max(1, Math.round(performance.now() - started)),
    backend: "extractive",
    analyzer: null,
    window: null,
    sufficiency: ranked.length > 0 ? { sufficient: ranked[0].score >= 0.34, score: ranked[0]?.score ?? 0 } : { sufficient: false, score: 0 },
    evidence: ranked.map(toEvidence),
  };
}

async function searchArticles(query: string): Promise<Article[]> {
  const url = new URL(API);
  url.searchParams.set("query", `${keywords(query)} sourcelang:english`);
  url.searchParams.set("mode", "ArtList");
  url.searchParams.set("format", "json");
  url.searchParams.set("maxrecords", "50");
  url.searchParams.set("timespan", "3m");
  url.searchParams.set("sort", "HybridRel");
  const response = await gdeltGet(url.toString());
  const text = response.text;
  if (response.status === 429 || /limit requests/i.test(text)) {
    throw new Error("GDELT is busy. Try again in a few seconds.");
  }
  if (response.status < 200 || response.status >= 300 || !text.trim().startsWith("{")) {
    throw new Error("GDELT could not be reached.");
  }
  const body = JSON.parse(text) as { articles?: Article[] };
  return (body.articles ?? []).filter((article) => article.title && article.url);
}

function gdeltGet(url: string, timeout = 25_000, hops = 0): Promise<{ status: number; text: string }> {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { "User-Agent": AGENT, Accept: "application/json,text/html" }, timeout }, (res) => {
      const location = res.headers.location;
      if (res.statusCode && res.statusCode >= 300 && res.statusCode < 400 && location && hops < 3) {
        res.resume();
        resolve(gdeltGet(new URL(location, url).toString(), timeout, hops + 1));
        return;
      }
      const chunks: Buffer[] = [];
      let size = 0;
      res.on("data", (chunk: Buffer) => {
        size += chunk.length;
        if (size <= 120_000) chunks.push(chunk);
        else res.destroy();
      });
      res.on("end", () => resolve({ status: res.statusCode ?? 0, text: Buffer.concat(chunks).toString("utf8") }));
      res.on("error", () => resolve({ status: res.statusCode ?? 0, text: Buffer.concat(chunks).toString("utf8") }));
    });
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("GDELT could not be reached."));
    });
    req.on("error", () => reject(new Error("GDELT could not be reached.")));
  });
}

function keywords(query: string): string {
  const terms = query
    .replace(/[^a-z0-9]+/gi, " ")
    .split(/\s+/)
    .filter((term) => term.length > 1 && !STOP.has(term.toLowerCase()));
  return terms.join(" ") || query;
}

function toEvidence(article: Article): Evidence {
  const where = [article.domain, article.sourcecountry].filter(Boolean).join(" · ");
  return {
    docId: `gdelt-${article.url}`,
    timestamp: seenDay(article.seendate ?? ""),
    title: article.title,
    entity: "GDELT",
    text: where || article.language || "GDELT news article",
    score: article.score ?? 0,
    url: article.url,
  };
}

function relevance(terms: string[], title: string): number {
  if (terms.length === 0) return 0;
  const text = title.toLowerCase();
  const hits = terms.filter((term) => text.includes(term)).length;
  return Number((hits / terms.length).toFixed(3));
}

function pointed(title: string, lead: string): string {
  const first = lead.split(/(?<=[.!?])\s+(?=[A-Z])/)[0]?.replace(/\s+/g, " ").trim() ?? "";
  if (first.length < 40) return title;
  const short = first.length > 220 ? `${first.slice(0, 200).replace(/\s+\S*$/, "")}…` : first;
  if (short.toLowerCase().startsWith(title.toLowerCase().slice(0, 28))) return short;
  return `${title}. ${short}`;
}

async function articleLead(url: string): Promise<string> {
  try {
    const response = await gdeltGet(url, 8000);
    if (response.status < 200 || response.status >= 300) return "";
    const decoded = decode(response.text);
    const description = decoded.match(/<meta[^>]+(?:name|property)=["'](?:og:description|description)["'][^>]+content=["']([^"']+)["']/i)
      ?? decoded.match(/<meta[^>]+content=["']([^"']+)["'][^>]+(?:name|property)=["'](?:og:description|description)["']/i);
    const text = (description?.[1] ?? "").replace(/\s+/g, " ").trim();
    return text.length > 40 ? text : "";
  } catch {
    return "";
  }
}

function decode(value: string): string {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, "\"")
    .replace(/&#39;|&apos;/g, "'")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">");
}

function seenDay(seendate: string): string {
  const match = seendate.match(/^(\d{4})(\d{2})(\d{2})/);
  return match ? `${match[1]}-${match[2]}-${match[3]}` : "";
}
