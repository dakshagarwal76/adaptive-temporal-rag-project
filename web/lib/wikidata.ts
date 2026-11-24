import { ask, type AskResult, type Evidence, type SystemId } from "@/lib/engine";

const API = "https://www.wikidata.org/w/api.php";
const AGENT = "AdaptiveTemporalRAG/1.0 (https://adaptive-temporal-rag.vercel.app)";
const STOP = new Set(["who", "what", "which", "when", "where", "whom", "whose", "whos", "the", "a", "an", "of", "is", "are", "was", "were", "did", "does", "how", "for", "and", "current", "currently", "now", "today"]);

type Snak = {
  snaktype?: string;
  datavalue?: { type?: string; value?: unknown };
};

type Claim = {
  rank?: string;
  mainsnak?: Snak;
  qualifiers?: Record<string, Snak[]>;
};

type Entity = {
  id?: string;
  labels?: { en?: { value?: string } };
  descriptions?: { en?: { value?: string } };
  claims?: Record<string, Claim[]>;
};

type Item = {
  id: string;
  title: string;
  description: string;
  text: string;
  timestamp: string;
  score: number;
};

export async function askWikidata(query: string, system: SystemId): Promise<AskResult> {
  const started = performance.now();
  if (system === "closed") {
    const closed = ask(query, "closed");
    return { ...closed, source: "wikidata" };
  }
  const items = await searchItems(query);
  const terms = queryTerms(query);
  const ranked = items
    .map((item) => ({ ...item, score: relevance(terms, item.title, `${item.description} ${item.text}`) }))
    .sort((a, b) => b.score - a.score || titlePrecision(terms, b.title) - titlePrecision(terms, a.title));
  return {
    query,
    system,
    source: "wikidata",
    answer: ranked[0] ? ranked[0].text : `No Wikidata item matched “${query}”.`,
    didRetrieve: ranked.length > 0,
    latencyMs: Math.max(1, Math.round(performance.now() - started)),
    backend: "extractive",
    analyzer: null,
    window: null,
    sufficiency: ranked.length > 0 ? { sufficient: ranked[0].score >= 0.34, score: ranked[0]?.score ?? 0 } : { sufficient: false, score: 0 },
    evidence: ranked.map(toEvidence),
  };
}

async function searchItems(query: string): Promise<Item[]> {
  const phrase = searchText(query);
  const [found, pages] = await Promise.all([
    wikidataGet({
      action: "wbsearchentities",
      search: phrase,
      language: "en",
      uselang: "en",
      type: "item",
      limit: "20",
    }),
    wikidataGet({
      action: "query",
      list: "search",
      srsearch: phrase,
      srlimit: "20",
      srnamespace: "0",
    }),
  ]);
  const labelIds = ((found.search ?? []) as Array<{ id?: string }>).map((hit) => hit.id).filter((id): id is string => Boolean(id));
  const textIds = ((pages.query?.search ?? []) as Array<{ title?: string }>).map((hit) => hit.title || "").filter((id) => /^Q\d+$/.test(id));
  const ids = [...new Set([...labelIds, ...textIds])].slice(0, 20);
  if (ids.length === 0) return [];
  const described = await wikidataGet({
    action: "wbgetentities",
    ids: ids.join("|"),
    props: "labels|descriptions",
    languages: "en",
  });
  const entities = { ...(described.entities ?? {}) } as Record<string, Entity>;
  const withClaims = ids.slice(0, 8);
  const detailed = await wikidataGet({
    action: "wbgetentities",
    ids: withClaims.join("|"),
    props: "claims",
    languages: "en",
  });
  for (const [id, entity] of Object.entries(detailed.entities ?? {})) {
    entities[id] = { ...entities[id], claims: entity.claims };
  }
  const terms = queryTerms(query);
  const chosen = new Map<string, Array<{ property: string; claim: Claim }>>();
  const needed = new Set<string>();
  for (const id of withClaims) {
    const claims = candidateClaims(entities[id]?.claims ?? {});
    chosen.set(id, claims);
    for (const row of claims) {
      needed.add(row.property);
      collectIds(row.claim.mainsnak, needed);
      for (const [property, group] of Object.entries(row.claim.qualifiers ?? {})) {
        needed.add(property);
        for (const snak of group.slice(0, 2)) collectIds(snak, needed);
      }
    }
  }
  const labels = await labelsFor([...needed]);
  for (const entity of Object.values(entities)) {
    const name = entity.labels?.en?.value;
    if (entity.id && name) labels.set(entity.id, name);
  }
  return ids.flatMap((id) => {
    const entity = entities[id];
    const title = entity?.labels?.en?.value || labels.get(id) || id;
    const description = entity?.descriptions?.en?.value || "";
    const rows = rankClaims(chosen.get(id) ?? [], labels, terms, title);
    const statements = rows.map((row) => renderClaim(row.property, row.claim, labels)).filter(Boolean);
    const text = [description, ...statements].filter(Boolean).join(" ");
    if (!text) return [];
    return [{ id, title, description, text, timestamp: rows.map((row) => claimTime(row.claim)).find(Boolean) || "", score: 0 }];
  });
}

async function wikidataGet(params: Record<string, string>): Promise<{ error?: { info?: string }; search?: unknown[]; query?: { search?: Array<{ title?: string }> }; entities?: Record<string, Entity> }> {
  const url = new URL(API);
  url.searchParams.set("format", "json");
  url.searchParams.set("utf8", "1");
  for (const [key, value] of Object.entries(params)) url.searchParams.set(key, value);
  const response = await fetch(url, {
    headers: { "User-Agent": AGENT, Accept: "application/json" },
    cache: "no-store",
    signal: AbortSignal.timeout(20000),
  });
  if (!response.ok) throw new Error(`Wikidata returned ${response.status}`);
  const body = await response.json();
  if (body.error) throw new Error(body.error.info || "Wikidata rejected the request.");
  return body;
}

async function labelsFor(ids: string[]): Promise<Map<string, string>> {
  const labels = new Map<string, string>();
  const unique = [...new Set(ids)].filter((id) => /^[QP]\d+$/.test(id));
  for (let start = 0; start < unique.length; start += 50) {
    const body = await wikidataGet({
      action: "wbgetentities",
      ids: unique.slice(start, start + 50).join("|"),
      props: "labels",
      languages: "en",
    });
    for (const [id, entity] of Object.entries(body.entities ?? {})) {
      const name = entity.labels?.en?.value;
      if (name) labels.set(id, name);
    }
  }
  return labels;
}

function candidateClaims(claims: Record<string, Claim[]>): Array<{ property: string; claim: Claim }> {
  const rows: Array<{ property: string; claim: Claim }> = [];
  for (const [property, list] of Object.entries(claims)) {
    const usable = list.filter((claim) => claim.rank !== "deprecated" && claim.mainsnak?.snaktype === "value");
    const preferred = usable.filter((claim) => claim.rank === "preferred");
    const pool = preferred.length > 0 ? preferred : usable;
    const timed = pool.filter(hasTime);
    const source = (timed.length > 0 ? timed : pool).slice().sort((a, b) => claimTime(b).localeCompare(claimTime(a)));
    for (const claim of source.slice(0, 2)) rows.push({ property, claim });
  }
  return rows;
}

function rankClaims(rows: Array<{ property: string; claim: Claim }>, labels: Map<string, string>, terms: string[], title: string) {
  const score = (row: { property: string; claim: Claim }) => {
    const prop = (labels.get(row.property) || "").toLowerCase();
    const value = snakText(row.claim.mainsnak, labels).toLowerCase();
    const overlap = echoes(value, title) ? 0 : terms.filter((term) => prop.includes(term) || value.includes(term)).length;
    const entity = row.claim.mainsnak?.datavalue?.type === "wikibase-entityid" ? 2 : 0;
    return overlap * 5 + entity + (hasTime(row.claim) ? 2 : 0) + (row.claim.rank === "preferred" ? 1 : 0);
  };
  const ranked = rows.slice().sort((a, b) => score(b) - score(a));
  const holder = ranked.find((row) => {
    const value = snakText(row.claim.mainsnak, labels);
    return row.claim.mainsnak?.datavalue?.type === "wikibase-entityid" && hasTime(row.claim) && !echoes(value, title);
  });
  const rest = holder ? ranked.filter((row) => row !== holder) : ranked;
  return (holder ? [holder, ...rest] : ranked).slice(0, 8);
}

function echoes(value: string, title: string): boolean {
  const words = title.toLowerCase().split(/[^a-z0-9]+/).filter((word) => word.length > 3);
  if (words.length === 0 || !value) return false;
  const blob = value.toLowerCase();
  return words.filter((word) => blob.includes(word)).length / words.length >= 0.6;
}

function hasTime(claim: Claim): boolean {
  return [claim.mainsnak, ...Object.values(claim.qualifiers ?? {}).flat()].some((snak) => snak?.datavalue?.type === "time");
}

function claimTime(claim: Claim): string {
  const times = [claim.mainsnak, ...Object.values(claim.qualifiers ?? {}).flat()].map(timeOf).filter(Boolean).sort();
  return times.at(-1) ?? "";
}

function collectIds(snak: Snak | undefined, ids: Set<string>) {
  const value = snak?.datavalue?.value;
  if (snak?.datavalue?.type === "wikibase-entityid" && value && typeof value === "object" && "id" in value) {
    ids.add(String(value.id));
  }
}

function renderClaim(property: string, claim: Claim, labels: Map<string, string>): string {
  const value = snakText(claim.mainsnak, labels);
  if (!value) return "";
  const qualifiers = Object.entries(claim.qualifiers ?? {})
    .flatMap(([name, group]) => group.map((snak) => {
      const text = snakText(snak, labels);
      return text ? `${labels.get(name) || name} ${text}` : "";
    }))
    .filter(Boolean)
    .slice(0, 3);
  const extra = qualifiers.length > 0 ? ` (${qualifiers.join(", ")})` : "";
  return `${labels.get(property) || property}: ${value}${extra}.`;
}

function snakText(snak: Snak | undefined, labels: Map<string, string>): string {
  if (!snak || snak.snaktype === "novalue") return "";
  if (snak.snaktype === "somevalue") return "unknown value";
  const value = snak.datavalue?.value;
  const type = snak.datavalue?.type;
  if (type === "wikibase-entityid" && value && typeof value === "object" && "id" in value) return labels.get(String(value.id)) || String(value.id);
  if (type === "time" && value && typeof value === "object" && "time" in value) return formatTime(String(value.time));
  if (type === "string" && typeof value === "string") return value;
  if (type === "monolingualtext" && value && typeof value === "object" && "text" in value) return String(value.text);
  if (type === "quantity" && value && typeof value === "object" && "amount" in value) return String(value.amount).replace(/^\+/, "");
  return "";
}

function timeOf(snak: Snak | undefined): string {
  if (snak?.datavalue?.type !== "time") return "";
  const value = snak.datavalue.value;
  if (!value || typeof value !== "object" || !("time" in value)) return "";
  return formatTime(String(value.time));
}

function formatTime(time: string): string {
  const match = time.match(/([+-]?\d{4})-(\d{2})-(\d{2})/);
  if (!match) return time.replace(/^\+/, "");
  const year = match[1].replace(/^\+/, "");
  if (match[2] === "00") return year;
  if (match[3] === "00") return `${year}-${match[2]}`;
  return `${year}-${match[2]}-${match[3]}`;
}

function toEvidence(item: Item): Evidence {
  return {
    docId: `wd-${item.id}`,
    timestamp: item.timestamp,
    title: item.title,
    entity: "Wikidata",
    text: item.text,
    score: item.score,
    url: `https://www.wikidata.org/wiki/${item.id}`,
  };
}

function searchText(query: string): string {
  const cleaned = query
    .replace(/[?]/g, " ")
    .replace(/^(who|what|which|when|where|whom|whose|how)\b/i, " ")
    .replace(/\b(is|are|was|were|current|currently)\b/gi, " ")
    .replace(/^\s*(the|a|an)\b/i, " ")
    .replace(/\s+/g, " ")
    .trim();
  return cleaned || query;
}

function queryTerms(query: string): string[] {
  return [...new Set(query.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/).filter((term) => term.length > 1 && !STOP.has(term)))];
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
