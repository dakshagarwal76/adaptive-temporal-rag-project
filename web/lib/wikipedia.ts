import { ask, type AskResult, type Evidence, type SystemId } from "@/lib/engine";

const API = "https://en.wikipedia.org/w/api.php";
const AGENT = "AdaptiveTemporalRAG/1.0 (https://adaptive-temporal-rag.vercel.app)";
const STOP = new Set(["who", "what", "which", "when", "where", "whom", "whose", "whos", "the", "a", "an", "of", "is", "are", "was", "were", "did", "does", "how", "for", "and", "current", "currently", "now", "today"]);
const FUNCTION = new Set(["a", "an", "the", "of", "and", "for", "in", "on", "to"]);

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
  const aliases = aliasesFromTitles(query, pages.map((page) => page.title));
  const terms = queryTerms(query);
  const ranked = pages
    .map((page) => ({ ...page, score: relevance(terms, aliases, page.title, page.extract) }))
    .sort((a, b) => b.score - a.score || titlePrecision(terms, aliases, b.title) - titlePrecision(terms, aliases, a.title) || listRank(a.title) - listRank(b.title));
  return {
    query,
    system,
    source: "wikipedia",
    answer: answerFrom(query, terms, aliases, ranked),
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

function answerFrom(query: string, terms: string[], aliases: Aliases, pages: WikiPage[]): string {
  if (pages.length === 0) return `No Wikipedia page matched “${query}”.`;
  const intent = intentOf(query);
  if (intent.definition) {
    const lead = sentences(pages[0].extract).find((sentence) => mentionsSubject(sentence, terms, aliases));
    if (lead) return lead.length <= 400 ? lead : clip(lead, 320);
  }
  const pool = pages.filter((page) => listRank(page.title) === 0).slice(0, 15);
  const source = pool.length > 0 ? pool : pages.slice(0, 1);
  const ranked = source.flatMap((page, pageIndex) =>
    sentences(page.extract).map((sentence, sentenceIndex) => ({
      sentence,
      title: page.title,
      score: sentenceScore(sentence, terms, aliases, intent, pageIndex, sentenceIndex, query, page.title),
    })),
  );
  ranked.sort((a, b) => b.score - a.score);
  const officeWords = terms.filter((term) => term.length > 3);
  const dated = intent.year
    ? ranked.find((item) => item.sentence.includes(intent.year ?? "") && officeWords.every((word) => hasWord(item.sentence, word)))
    : undefined;
  if (intent.year && !dated) return `The matching Wikipedia leads do not say who held that office in ${intent.year}.`;
  const grounded = ranked.filter((item) => item.score > 0 && mentionsSubject(item.sentence, terms, aliases) && !/\b(compared to|similar to|such as|for example)\b/i.test(item.sentence));
  const direct = grounded.filter((item) => holdsOffice(item.sentence));
  const best = dated ?? direct[0] ?? grounded[0] ?? ranked.find((item) => item.score > 0);
  const sentence = best?.sentence ?? sentences(source[0].extract)[0] ?? source[0].extract;
  return toThePoint(sentence, intent);
}

type Intent = { who: boolean; historical: boolean; definition: boolean; year: string | null };

function intentOf(query: string): Intent {
  return {
    who: /\bwho(?:m|se|'s|s)?\b/i.test(query),
    historical: /\b(was|were|former|previous|used to)\b/i.test(query) || /\b(?:19|20)\d{2}\b/.test(query),
    definition: /\b(what(?:'s| is)|define|explain|meaning)\b/i.test(query),
    year: query.match(/\b(?:19|20)\d{2}\b/)?.[0] ?? null,
  };
}

function sentenceScore(sentence: string, terms: string[], aliases: Aliases, intent: Intent, pageIndex: number, sentenceIndex: number, query: string, title: string): number {
  const hits = overlap(terms, aliases, sentence);
  if (hits === 0 && !(intent.definition && pageIndex === 0 && sentenceIndex === 0)) return 0;
  let score = hits * 2 + Math.max(0, 2 - pageIndex * 0.2);
  if (intent.definition && pageIndex === 0 && sentenceIndex === 0) score += 6;
  if (intent.who) {
    const name = personName(sentence, query, title);
    const copula = Boolean(name && new RegExp(`\\b(is|was|serves as|served as|has been)\\b[^.]{0,60}${escapeRegExp(name)}|${escapeRegExp(name)}\\b[^.]{0,90}\\b(is|was|serves as|served as)\\b`, "i").test(sentence));
    if (copula && !intent.historical && /\b(is|serves as|has been)\b/i.test(sentence)) score += 6;
    else if (copula) score += intent.historical ? 5 : 2;
    else if (name) score += 0.5;
    else score -= 2;
    if (!intent.historical && /\bcurrent(?:ly)?\b/i.test(sentence)) score += 4;
    if (name && mentionsSubject(sentence, terms, aliases) && holdsOffice(sentence)) score += 7;
    if (!intent.historical && /\b(former|previously)\b/i.test(sentence)) score -= 3;
    if (intent.year && sentence.includes(intent.year)) score += 4;
  }
  if (sentence.length > 320) score -= 0.5;
  return score;
}

function holdsOffice(sentence: string): boolean {
  return /\b(acted as|served as|serves as|was appointed|appointed as|became)\b[^.]{0,40}\b(ceo|chief executive|prime minister|president)\b/i.test(sentence)
    || /\b(is|was)\s+the\s+(current\s+)?(ceo|chief executive|prime minister|president)\b/i.test(sentence)
    || /\b(ceo|chief executive|prime minister|president)\s+is\b/i.test(sentence)
    || /\bwho is the current\b/i.test(sentence);
}

function toThePoint(sentence: string, intent: Intent): string {
  const clean = sentence.replace(/\s+/g, " ").trim();
  if (!intent.who) return clean.length <= 400 ? clean : clip(clean, 320);
  const holder = clean.match(/([A-Z][a-z]+(?:\s+[A-Z][a-z.'-]+){1,2}), who is the current [^.]{0,100}?since\s+(?:\d{1,2}\s+[A-Z][a-z]+\s+)?\d{4}/);
  if (holder) {
    const text = holder[0].replace(/,\s*who\s+/i, " ").trim();
    return /[.!?]$/.test(text) ? text : `${text}.`;
  }
  if (clean.length <= 240) return clean;
  return clip(clean, 240);
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function personName(sentence: string, query: string, title: string): string | null {
  const blocked = new Set(`${query} ${title}`.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/).filter((word) => word.length > 1));
  const matches = sentence.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z.'-]+){1,3}\b/g) ?? [];
  const people = matches.filter((match) => {
    const parts = match.split(/\s+/);
    if (MONTHS.has(parts[0])) return false;
    return parts.some((part) => !blocked.has(part.toLowerCase()));
  });
  if (people.length === 0) return null;
  const focusAt = focusIndex(sentence);
  return people.sort((a, b) => Math.abs(sentence.indexOf(a) - focusAt) - Math.abs(sentence.indexOf(b) - focusAt))[0];
}

function focusIndex(sentence: string): number {
  const office = sentence.search(/\b(chief executive(?: officer)?|ceo|prime minister|president)\b/i);
  if (office >= 0) return office;
  const current = sentence.search(/\bcurrent(?:ly)?\b/i);
  if (current >= 0) return current;
  const copula = sentence.search(/\b(is|was|serves as|served as)\b/i);
  return copula >= 0 ? copula : 0;
}

function mentionsSubject(sentence: string, terms: string[], aliases: Aliases): boolean {
  const needed = terms.filter((term) => term.length > 2);
  if (needed.length === 0) return true;
  return needed.every((term) => hasWord(sentence, term) || aliasCovers(sentence, aliases.get(term)));
}

function aliasCovers(sentence: string, words: string[] | undefined): boolean {
  return Boolean(words && words.length > 0 && words.every((word) => hasWord(sentence, word)));
}

function clip(text: string, max: number): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max).replace(/\s+\S*$/, "")}…`;
}

const MONTHS = new Set(["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]);

function listRank(title: string): number {
  return /^list of\b/i.test(title) ? 1 : 0;
}

type Aliases = Map<string, string[]>;

function aliasesFromTitles(query: string, titles: string[]): Aliases {
  const counts = new Map<string, Map<string, number>>();
  for (const title of titles) {
    const words = title.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/).filter((word) => word.length >= 3 && !FUNCTION.has(word) && !STOP.has(word));
    const seen = new Set<string>();
    for (let size = 2; size <= 3; size += 1) {
      for (let index = 0; index + size <= words.length; index += 1) {
        const window = words.slice(index, index + size);
        const key = window.map((word) => word[0]).join("");
        const phrase = window.join(" ");
        if (seen.has(`${key}:${phrase}`)) continue;
        seen.add(`${key}:${phrase}`);
        const bucket = counts.get(key) ?? new Map<string, number>();
        bucket.set(phrase, (bucket.get(phrase) ?? 0) + 1);
        counts.set(key, bucket);
      }
    }
  }
  const aliases: Aliases = new Map();
  for (const token of query.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/)) {
    if (token.length < 2 || token.length > 4 || STOP.has(token)) continue;
    const bucket = counts.get(token);
    if (!bucket) continue;
    const ranked = [...bucket.entries()].sort((a, b) => b[1] - a[1]);
    if (ranked.length > 1 && ranked[0][1] === ranked[1][1]) continue;
    aliases.set(token, ranked[0][0].split(" "));
  }
  return aliases;
}

function covers(term: string, text: string, aliases: Aliases): boolean {
  if (hasWord(text, term)) return true;
  const words = aliases.get(term);
  return Boolean(words && words.every((word) => hasWord(text, word)));
}

function hasWord(text: string, word: string): boolean {
  return new RegExp(`\\b${escapeRegExp(word)}\\b`, "i").test(text);
}

function sentences(extract: string): string[] {
  const parts: string[] = [];
  let start = 0;
  for (let index = 0; index < extract.length; index += 1) {
    if (!/[.!?]/.test(extract[index] ?? "")) continue;
    const previous = extract[index - 1] ?? "";
    const earlier = extract[index - 2] ?? "";
    if (/[A-Z]/.test(previous) && !/[A-Za-z]/.test(earlier)) continue;
    const next = extract.slice(index + 1);
    if (!/^\s+[A-Z]/.test(next) && next.trim().length > 0) continue;
    const sentence = extract.slice(start, index + 1).trim();
    if (sentence.length > 40) parts.push(sentence);
    start = index + 1;
  }
  const tail = extract.slice(start).trim();
  if (tail.length > 40) parts.push(tail);
  return parts;
}

function queryTerms(query: string): string[] {
  return [...new Set(query.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/).filter((term) => term.length > 1 && !STOP.has(term)))];
}

function titlePrecision(terms: string[], aliases: Aliases, title: string): number {
  const words = title.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/).filter((word) => word.length > 2 && !STOP.has(word));
  if (words.length === 0) return 0;
  const hits = words.filter((word) => terms.some((term) => word.includes(term) || (aliases.get(term) ?? []).includes(word))).length;
  return hits / words.length;
}

function relevance(terms: string[], aliases: Aliases, title: string, text: string): number {
  if (terms.length === 0) return 0;
  const blob = `${title} ${text}`.toLowerCase();
  const titleText = title.toLowerCase();
  const hits = terms.filter((term) => covers(term, blob, aliases)).length;
  const titleHits = terms.filter((term) => covers(term, titleText, aliases)).length;
  return Number(((0.7 * hits + 0.3 * titleHits) / terms.length).toFixed(3));
}

function overlap(terms: string[], aliases: Aliases, sentence: string): number {
  const text = sentence.toLowerCase();
  return terms.reduce((hits, term) => hits + (covers(term, text, aliases) ? 1 : 0), 0);
}
