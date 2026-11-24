import part0 from "../data/corpus/part-0.json";
import part1 from "../data/corpus/part-1.json";
import part2 from "../data/corpus/part-2.json";
import part3 from "../data/corpus/part-3.json";

export type Doc = {
  doc_id: string;
  text: string;
  timestamp: string;
  title: string;
  source: string;
  entity: string;
  topic: string;
};

export type SystemId = "adaptive" | "standard" | "fixed" | "closed";

export type SourceId = "corpus" | "wikipedia" | "gdelt";

export type Evidence = {
  docId: string;
  timestamp: string;
  title: string;
  entity: string;
  text: string;
  score: number;
  url?: string;
};

export type AskResult = {
  query: string;
  system: SystemId;
  source: SourceId;
  answer: string;
  didRetrieve: boolean;
  latencyMs: number;
  backend: "extractive";
  analyzer: {
    needsRetrieval: boolean;
    temporalClass: "current" | "historical" | "timeless";
    confidence: number;
    policy: string;
  } | null;
  window: {
    targetYear: number | null;
    initialYears: number;
    finalYears: number;
    reason: string;
  } | null;
  sufficiency: { sufficient: boolean; score: number } | null;
  evidence: Evidence[];
};

const DOCS = [...part0, ...part1, ...part2, ...part3] as Doc[];
const REFERENCE = new Date(Date.UTC(2025, 9, 1));
const TOP_K = 20;

const STOP = new Set([
  "the", "and", "for", "who", "what", "which", "when", "where", "whom", "whose",
  "did", "does", "was", "were", "from", "between", "before", "after", "until",
  "since", "during", "year", "with", "into", "that", "this", "are", "you",
  "current", "currently", "now", "today", "please", "tell", "name", "serving",
  "most", "recent", "latest", "define", "explain", "how", "can", "about",
]);

const TRIVIAL: Array<[string[], string]> = [
  [["2 + 2", "2+2"], "4"],
  [["10 times 5"], "50"],
  [["7 + 8", "7+8"], "15"],
  [["days are in a week"], "7"],
  [["hours are in a day"], "24"],
  [["minutes are in an hour"], "60"],
  [["h2o"], "Yes — water is H2O."],
  [["opposite of hot"], "cold"],
  [["sun a star"], "Yes"],
  [["clear daytime sky"], "blue"],
  [["spell the word cat"], "cat"],
  [["angles in a triangle"], "180"],
];

type Scored = { doc: Doc; score: number; index: number };

const FUNCTION_WORDS = new Set(["a", "an", "the", "of", "and", "for", "in", "on", "to", "de", "du", "von", "di"]);

const index = buildIndex(DOCS);
const namePhrases = buildNamePhrases(DOCS);

function foldAbbrev(text: string): string {
  let current = text.toLowerCase();
  let previous = "";
  while (current !== previous) {
    previous = current;
    current = current.replace(/\b([a-z])\.(?=[a-z])/g, "$1");
  }
  return current;
}

function tokens(text: string): string[] {
  return foldAbbrev(text)
    .replace(/[^a-z0-9]+/g, " ")
    .split(/\s+/)
    .filter((token) => token.length > 2 && !STOP.has(token) && !/^(19|20)\d{2}$/.test(token));
}

function nameWords(text: string): string[] {
  return foldAbbrev(text)
    .replace(/[^a-z0-9]+/g, " ")
    .split(/\s+/)
    .filter((word) => word.length >= 3 && !FUNCTION_WORDS.has(word) && !STOP.has(word) && !/^(19|20)\d{2}$/.test(word));
}

function buildNamePhrases(docs: Doc[]): Map<string, Set<string>> {
  const phrases = new Map<string, Set<string>>();
  for (const entity of new Set(docs.map((doc) => doc.entity).filter(Boolean))) {
    const words = nameWords(entity);
    for (let start = 0; start < words.length; start++) {
      let letters = "";
      for (let end = start; end < words.length && end < start + 4; end++) {
        letters += words[end][0];
        if (end === start) continue;
        const key = letters;
        const phrase = words.slice(start, end + 1).join(" ");
        const known = phrases.get(key) ?? new Set<string>();
        known.add(phrase);
        phrases.set(key, known);
      }
    }
  }
  return phrases;
}

function queryTokens(text: string): string[] {
  const base = tokens(text);
  const seen = new Set(base);
  const pieces = foldAbbrev(text)
    .replace(/[^a-z0-9]+/g, " ")
    .split(/\s+/)
    .filter((piece) => piece.length >= 2 && piece.length <= 6 && !STOP.has(piece) && !FUNCTION_WORDS.has(piece));
  for (const piece of pieces) {
    if (seen.has(piece)) continue;
    const phrases = namePhrases.get(piece);
    if (!phrases || phrases.size !== 1) continue;
    for (const word of [...phrases][0].split(" ")) {
      if (seen.has(word) || word.length <= 2) continue;
      seen.add(word);
      base.push(word);
    }
  }
  return base;
}

function isWhoQuestion(query: string): boolean {
  return /\bwho(?:m|se|'s|s)?\b/i.test(query);
}

function buildIndex(docs: Doc[]) {
  const docsTokens = docs.map((doc) => tokens(`${doc.entity} ${doc.title} ${doc.text}`));
  const df = new Map<string, number>();
  let total = 0;
  for (const row of docsTokens) {
    total += row.length;
    for (const token of new Set(row)) df.set(token, (df.get(token) ?? 0) + 1);
  }
  const avg = total / Math.max(docs.length, 1);
  const idf = new Map<string, number>();
  for (const [token, count] of df) {
    idf.set(token, Math.log(1 + (docs.length - count + 0.5) / (count + 0.5)));
  }
  return { docsTokens, idf, avg };
}

function bm25(query: string): Scored[] {
  const q = queryTokens(query);
  const k1 = 1.2;
  const b = 0.75;
  const scored: Scored[] = [];
  for (let i = 0; i < DOCS.length; i++) {
    const row = index.docsTokens[i];
    if (q.length === 0 || row.length === 0) continue;
    const tf = new Map<string, number>();
    for (const token of row) tf.set(token, (tf.get(token) ?? 0) + 1);
    let score = 0;
    for (const token of q) {
      const freq = tf.get(token);
      const weight = index.idf.get(token);
      if (!freq || weight === undefined) continue;
      const norm = 1 - b + b * (row.length / index.avg);
      score += weight * ((freq * (k1 + 1)) / (freq + k1 * norm));
    }
    if (score > 0) scored.push({ doc: DOCS[i], score, index: i });
  }
  scored.sort((a, b) => b.score - a.score || a.index - b.index);
  return scored;
}

function yearOf(timestamp: string): number {
  return Number(timestamp.slice(0, 4));
}

function dateOf(timestamp: string): Date {
  const [y, m, d] = timestamp.split("-").map(Number);
  return new Date(Date.UTC(y, (m || 1) - 1, d || 1));
}

function extractYear(query: string): number | null {
  const cleaned = query.replace(/\([^)]*\bborn\b[^)]*\)/gi, " ");
  const match = cleaned.match(/\b(19|20)\d{2}\b/);
  return match ? Number(match[0]) : null;
}

function isTrivial(query: string): boolean {
  const q = query.toLowerCase();
  return TRIVIAL.some(([keys]) => keys.some((key) => q.includes(key)));
}

function trivialAnswer(query: string): string | null {
  const q = query.toLowerCase();
  for (const [keys, answer] of TRIVIAL) {
    if (keys.some((key) => q.includes(key))) return answer;
  }
  return null;
}

function isDefinition(query: string): boolean {
  const q = query.toLowerCase().trim();
  return /^(what is|what's|define|explain)\b/.test(q);
}

type Analysis = {
  needsRetrieval: number;
  temporalClass: "current" | "historical" | "timeless";
  confidence: number;
};

function analyze(query: string): Analysis {
  if (isTrivial(query)) {
    return { needsRetrieval: 0, temporalClass: "timeless", confidence: 0.94 };
  }
  const year = extractYear(query);
  const q = foldAbbrev(query);
  const currentCue =
    /\b(current|currently|nowadays|right now|today|at present)\b/.test(q) ||
    (isWhoQuestion(q) && year === null && !/\b(was|were|former|previous|used to)\b/.test(q));
  if (isDefinition(query) && year === null && !/\b(current|currently|today)\b/.test(q)) {
    return { needsRetrieval: 0, temporalClass: "timeless", confidence: 0.82 };
  }
  if (currentCue && year === null) {
    return { needsRetrieval: 1, temporalClass: "current", confidence: 0.9 };
  }
  if (year !== null || /\b(was|were|former|previous|used to)\b/.test(q)) {
    return { needsRetrieval: 1, temporalClass: "historical", confidence: 0.88 };
  }
  return { needsRetrieval: 1, temporalClass: "timeless", confidence: 0.72 };
}

function route(analysis: Analysis, query: string) {
  if (analysis.needsRetrieval === 0 && isTrivial(query)) {
    return { skip: true, policy: "no_retrieval", mode: "semantic" as const };
  }
  const soft = analysis.needsRetrieval === 0;
  if (soft || analysis.confidence < 0.55 || analysis.temporalClass === "timeless") {
    const policy = soft
      ? "soft_retrieve_timeless"
      : analysis.confidence < 0.55
        ? "low_conf_semantic"
        : "timeless_semantic";
    return { skip: false, policy, mode: "semantic" as const };
  }
  if (analysis.temporalClass === "current") {
    return { skip: false, policy: "current", mode: "current" as const };
  }
  return { skip: false, policy: "historical_adaptive_window", mode: "historical" as const };
}

function overlap(query: string, doc: Doc, entityOnly: boolean): number {
  const q = new Set(queryTokens(query));
  if (q.size === 0) return 0;
  const blob = (entityOnly ? doc.entity : `${doc.entity} ${doc.title} ${doc.text}`).toLowerCase();
  let hits = 0;
  for (const token of q) if (blob.includes(token)) hits += 1;
  return hits / q.size;
}

function entityPool(cands: Scored[], query: string): Scored[] {
  if (cands.length === 0) return cands;
  const best = Math.max(...cands.map((row) => overlap(query, row.doc, true)));
  if (best <= 0) return cands;
  const matched = cands.filter((row) => overlap(query, row.doc, true) >= best - 0.01);
  const entities = new Set(matched.map((row) => row.doc.entity).filter(Boolean));
  if (entities.size === 0) return matched;
  const byId = new Map(cands.map((row) => [row.doc.doc_id, row]));
  const expanded: Scored[] = [];
  DOCS.forEach((doc, i) => {
    if (!entities.has(doc.entity)) return;
    expanded.push(byId.get(doc.doc_id) ?? { doc, score: 0.01, index: i });
  });
  return expanded.length > 0 ? expanded : matched;
}

function inferWindow(query: string) {
  const year = extractYear(query);
  const decade = query.toLowerCase().match(/\b((?:19|20)\d)0s\b/);
  if (decade) {
    const y0 = Number(`${decade[1]}0`);
    return { targetYear: y0 + 5, initialYears: 5, maxYears: 8, reason: "decade" };
  }
  if (year !== null && new RegExp(`\\b(early|mid|late)\\b.*\\b${year}\\b|\\b${year}\\b.*(early|mid|late)`).test(query.toLowerCase())) {
    return { targetYear: year, initialYears: 1.5, maxYears: 3.5, reason: "year_phase" };
  }
  if (year !== null) {
    return { targetYear: year, initialYears: 1, maxYears: 4, reason: "year" };
  }
  return { targetYear: REFERENCE.getUTCFullYear() - 5, initialYears: 3, maxYears: 8, reason: "vague_historical" };
}

function within(doc: Doc, targetYear: number, halfYears: number): boolean {
  const target = Date.UTC(targetYear, 5, 15);
  const distance = Math.abs(dateOf(doc.timestamp).getTime() - target) / 86400000;
  return distance <= halfYears * 365.25;
}

function rank(pool: Scored[], mode: "current" | "historical" | "semantic", query: string, windowYears: number | null) {
  if (mode === "current") {
    return [...pool].sort((a, b) => {
      const byDate = dateOf(b.doc.timestamp).getTime() - dateOf(a.doc.timestamp).getTime();
      return byDate || b.score - a.score;
    });
  }
  if (mode === "semantic") {
    return [...pool].sort((a, b) => b.score - a.score || a.index - b.index);
  }
  const year = extractYear(query) ?? REFERENCE.getUTCFullYear() - 5;
  const halfLife = Math.max(90, (windowYears ?? 5) * 365.25);
  const target = Date.UTC(year, 5, 15);
  return [...pool].sort((a, b) => {
    const ta = 1 - Math.min(1, Math.abs(dateOf(a.doc.timestamp).getTime() - target) / 86400000 / halfLife);
    const tb = 1 - Math.min(1, Math.abs(dateOf(b.doc.timestamp).getTime() - target) / 86400000 / halfLife);
    return tb + 0.01 * b.score - (ta + 0.01 * a.score);
  });
}

function retrieve(query: string, mode: "current" | "historical" | "semantic") {
  const cands = bm25(query).slice(0, 80);
  const pool = entityPool(cands, query);
  let window: AskResult["window"] = null;
  let ranked = pool;
  if (mode === "historical") {
    const spec = inferWindow(query);
    let half = spec.initialYears;
    let kept: Scored[] = [];
    let expansions = 0;
    while (true) {
      kept = pool.filter((row) => spec.targetYear !== null && within(row.doc, spec.targetYear, half));
      if (kept.length > 0 || half >= spec.maxYears - 1e-9) break;
      const next = Math.min(spec.maxYears, half * 1.5 + 0.25);
      if (next <= half + 1e-9) break;
      half = next;
      expansions += 1;
    }
    ranked = kept.length > 0 ? kept : pool;
    window = {
      targetYear: spec.targetYear,
      initialYears: spec.initialYears,
      finalYears: Number(half.toFixed(2)),
      reason: expansions > 0 ? `${spec.reason}, expanded ${expansions}` : spec.reason,
    };
    ranked = rank(ranked, "historical", query, half);
  } else if (mode === "current") {
    ranked = rank(pool, "current", query, null);
  } else {
    ranked = rank(pool, "semantic", query, null);
  }
  return { ranked: ranked.slice(0, TOP_K), window };
}

function fixedRetrieve(query: string) {
  const cands = entityPool(bm25(query).slice(0, 80), query);
  const year = extractYear(query);
  if (year !== null) {
    const kept = cands.filter((row) => Math.abs(yearOf(row.doc.timestamp) - year) <= 2);
    const pool = kept.length > 0 ? kept : cands;
    return {
      ranked: rank(pool, "historical", query, 2).slice(0, TOP_K),
      window: { targetYear: year, initialYears: 2, finalYears: 2, reason: "fixed ±2 years" },
      mode: "historical" as const,
    };
  }
  return {
    ranked: rank(cands, "current", query, null).slice(0, TOP_K),
    window: null,
    mode: "current" as const,
  };
}

function project(query: string, answer: string): string {
  const text = answer.trim();
  if (!isWhoQuestion(query)) return text;
  const paren = text.match(/^(.+?)\s+\([^)]+\)$/);
  if (paren && !/\sof\s/i.test(paren[1])) return paren[1].trim();
  const identified = text.match(/\bidentify\s+(.+?)\s+as\b/i);
  if (identified) return identified[1].trim();
  const held = text.match(/^(.+?)\s+held the office\b/i);
  if (held) return held[1].trim();
  const office = text.match(/^(.+?)\s+assumed office\b/i);
  if (office) return office[1].replace(/^On \d{4}-\d{2}-\d{2},?\s*/i, "").trim();
  return text;
}

function readAnswer(query: string, docs: Doc[]): string {
  const trivial = trivialAnswer(query);
  if (trivial) return trivial;
  if (docs.length === 0) {
    const topic = query.replace(/^(what is|what's|define|explain)\s+/i, "").replace(/[?.]+$/g, "").trim();
    return topic
      ? `No evidence in the corpus for “${topic}”.`
      : "Not enough evidence to answer.";
  }
  const text = docs[0].text.trim();
  if (isDefinition(query)) return text;
  const holder = text.match(/(?:As of [^,]+,\s*)?([A-Z][\w .'-]+?)\s+(?:is|was)\s+the\s+(.+?)\./);
  if (holder) return project(query, `${holder[1].trim()} (${holder[2].trim()})`);
  const won = text.match(/([A-Z][\w .'-]+?)\s+won\b/);
  if (won) return won[1].trim();
  const identified = text.match(/\bidentify\s+([A-Z][\w .'-]+?)\s+as\b/);
  if (identified) return identified[1].trim();
  const assumed = text.match(/([A-Z][\w .'-]+?)\s+assumed office\b/);
  if (assumed) return assumed[1].trim();
  return project(query, text);
}

function sufficiency(query: string, docs: Doc[], mode: string): { sufficient: boolean; score: number } {
  const q = new Set(queryTokens(query));
  if (docs.length === 0 || q.size === 0) return { sufficient: false, score: 0 };
  const blob = docs.map((doc) => `${doc.entity} ${doc.title} ${doc.text}`.toLowerCase()).join(" ");
  let hits = 0;
  for (const token of q) if (blob.includes(token)) hits += 1;
  const coverage = hits / q.size;
  let aligned = 0.5;
  const year = extractYear(query);
  const topYear = yearOf(docs[0].timestamp);
  if (mode === "historical" && year !== null) aligned = Math.abs(topYear - year) <= 1 ? 1 : Math.abs(topYear - year) <= 2 ? 0.7 : 0.2;
  if (mode === "current") aligned = topYear >= 2023 ? 1 : topYear >= 2020 ? 0.55 : 0.25;
  const score = Number((0.6 * coverage + 0.4 * aligned).toFixed(3));
  return { sufficient: score >= 0.55, score };
}

function pack(rows: Scored[]): Evidence[] {
  return rows.map((row) => ({
    docId: row.doc.doc_id,
    timestamp: row.doc.timestamp,
    title: row.doc.title || row.doc.entity,
    entity: row.doc.entity,
    text: row.doc.text,
    score: Number(row.score.toFixed(3)),
  }));
}

export function ask(query: string, system: SystemId): AskResult {
  const started = performance.now();
  const q = query.trim();
  const analysis = analyze(q);

  if (system === "closed") {
    return {
      query: q,
      system,
      source: "corpus",
      answer: readAnswer(q, []),
      didRetrieve: false,
      latencyMs: Math.max(1, Math.round(performance.now() - started)),
      backend: "extractive",
      analyzer: null,
      window: null,
      sufficiency: null,
      evidence: [],
    };
  }

  if (system === "standard") {
    const { ranked } = retrieve(q, "semantic");
    const docs = ranked.map((row) => row.doc);
    return {
      query: q,
      system,
      source: "corpus",
      answer: readAnswer(q, docs),
      didRetrieve: true,
      latencyMs: Math.max(1, Math.round(performance.now() - started)),
      backend: "extractive",
      analyzer: null,
      window: null,
      sufficiency: sufficiency(q, docs, "semantic"),
      evidence: pack(ranked),
    };
  }

  if (system === "fixed") {
    const { ranked, window, mode } = fixedRetrieve(q);
    const docs = ranked.map((row) => row.doc);
    return {
      query: q,
      system,
      source: "corpus",
      answer: readAnswer(q, docs),
      didRetrieve: true,
      latencyMs: Math.max(1, Math.round(performance.now() - started)),
      backend: "extractive",
      analyzer: null,
      window,
      sufficiency: sufficiency(q, docs, mode),
      evidence: pack(ranked),
    };
  }

  const decision = route(analysis, q);
  if (decision.skip) {
    return {
      query: q,
      system,
      source: "corpus",
      answer: readAnswer(q, []),
      didRetrieve: false,
      latencyMs: Math.max(1, Math.round(performance.now() - started)),
      backend: "extractive",
      analyzer: {
        needsRetrieval: false,
        temporalClass: analysis.temporalClass,
        confidence: analysis.confidence,
        policy: decision.policy,
      },
      window: null,
      sufficiency: null,
      evidence: [],
    };
  }

  const { ranked, window } = retrieve(q, decision.mode);
  const docs = ranked.map((row) => row.doc);
  return {
    query: q,
    system,
    source: "corpus",
    answer: readAnswer(q, docs),
    didRetrieve: true,
    latencyMs: Math.max(1, Math.round(performance.now() - started)),
    backend: "extractive",
    analyzer: {
      needsRetrieval: analysis.needsRetrieval === 1,
      temporalClass: analysis.temporalClass,
      confidence: analysis.confidence,
      policy: decision.policy,
    },
    window,
    sufficiency: sufficiency(q, docs, decision.mode),
    evidence: pack(ranked),
  };
}

export const EXAMPLES = [
  "Who is the current Prime Minister of India?",
  "Who was the President of the United States in 2018?",
  "Define photosynthesis.",
  "What is 2 + 2?",
];
