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
};

export async function askGdelt(query: string, system: SystemId): Promise<AskResult> {
  const started = performance.now();
  if (system === "closed") {
    const closed = ask(query, "closed");
    return { ...closed, source: "gdelt" };
  }

  const articles = await searchArticles(query);
  const evidence = articles.map((article, index) => toEvidence(article, index, articles.length));
  return {
    query,
    system,
    source: "gdelt",
    answer: articles.length > 0 ? articles[0].title : `No GDELT article matched “${query}” in the last 3 months.`,
    didRetrieve: evidence.length > 0,
    latencyMs: Math.max(1, Math.round(performance.now() - started)),
    backend: "extractive",
    analyzer: null,
    window: null,
    sufficiency: evidence.length > 0 ? { sufficient: true, score: 1 } : { sufficient: false, score: 0 },
    evidence,
  };
}

async function searchArticles(query: string): Promise<Article[]> {
  const url = new URL(API);
  url.searchParams.set("query", `${keywords(query)} sourcelang:english`);
  url.searchParams.set("mode", "ArtList");
  url.searchParams.set("format", "json");
  url.searchParams.set("maxrecords", "5");
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
  return (body.articles ?? []).filter((article) => article.title && article.url).slice(0, 5);
}

function gdeltGet(url: string): Promise<{ status: number; text: string }> {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { "User-Agent": AGENT, Accept: "application/json" }, timeout: 25_000 }, (res) => {
      const chunks: Buffer[] = [];
      res.on("data", (chunk: Buffer) => chunks.push(chunk));
      res.on("end", () => resolve({ status: res.statusCode ?? 0, text: Buffer.concat(chunks).toString("utf8") }));
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

function toEvidence(article: Article, index: number, total: number): Evidence {
  const where = [article.domain, article.sourcecountry].filter(Boolean).join(" · ");
  return {
    docId: `gdelt-${index}-${article.url}`,
    timestamp: seenDay(article.seendate ?? ""),
    title: article.title,
    entity: "GDELT",
    text: where || article.language || "GDELT news article",
    score: Number((1 - index / Math.max(total, 1)).toFixed(3)),
    url: article.url,
  };
}

function seenDay(seendate: string): string {
  const match = seendate.match(/^(\d{4})(\d{2})(\d{2})/);
  return match ? `${match[1]}-${match[2]}-${match[3]}` : "";
}
