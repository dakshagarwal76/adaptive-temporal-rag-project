import type { Evidence } from "@/lib/engine";

type Provider = {
  name: string;
  apiKey: string;
  baseUrl: string;
  model: string;
  headers: Record<string, string>;
};

export type Generation = {
  answer: string;
  backend: string;
};

const SYSTEM = "You are a concise factual assistant. Ground the answer in the evidence. Reply with one or two complete sentences and include the relevant date from the evidence.";
const STOP = new Set(["who", "what", "which", "when", "where", "whom", "whose", "the", "a", "an", "of", "is", "are", "was", "were", "did", "does", "how", "for", "and", "current", "currently"]);

export async function generateAnswer(query: string, evidence: Evidence[]): Promise<Generation | null> {
  const mode = env("GENERATOR_BACKEND").trim().toLowerCase() || "api";
  if (mode === "extractive" || mode === "hf") return null;
  const providers = loadProviders();
  if (providers.length === 0) return null;
  for (const provider of providers) {
    try {
      const answer = await complete(provider, query, evidence);
      if (!answer) continue;
      return { answer, backend: `api:${provider.name}:${provider.model}` };
    } catch {
      continue;
    }
  }
  return null;
}

function loadProviders(): Provider[] {
  const named = list(env("LLM_PROVIDER_ORDER"));
  const found = new Set(named);
  for (const key of Object.keys(process.env)) {
    const match = key.match(/^([A-Z0-9]+)_API_KEY$/);
    if (match) found.add(match[1].toLowerCase());
  }
  const order = named.length > 0 ? [...named, ...[...found].filter((name) => !named.includes(name))] : [...found];
  const providers: Provider[] = [];
  for (const name of order) {
    const prefix = name.replace(/[^a-z0-9]/g, "_").toUpperCase();
    const apiKey = env(`${prefix}_API_KEY`);
    const baseUrl = env(`${prefix}_BASE_URL`);
    const model = env(`${prefix}_MODEL`);
    if (!apiKey || !baseUrl || !model) continue;
    const headers: Record<string, string> = {};
    const site = env(`${prefix}_SITE_URL`);
    const app = env(`${prefix}_APP_NAME`);
    if (site) headers["HTTP-Referer"] = site;
    if (app) headers["X-Title"] = app;
    providers.push({ name, apiKey, baseUrl: baseUrl.replace(/\/$/, ""), model, headers });
  }
  return providers;
}

function list(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
}

function env(name: string): string {
  return process.env[name] || "";
}

async function complete(provider: Provider, query: string, evidence: Evidence[]): Promise<string> {
  const messages = [
    { role: "system", content: SYSTEM },
    { role: "user", content: prompt(query, evidence) },
  ];
  let budget = numberEnv("LLM_MAX_TOKENS", 512);
  let answer = "";
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const result = await chat(provider, messages, budget);
    answer = result.text;
    if (result.finish !== "length" && /[.!?]["']?$/.test(answer)) return answer;
    budget *= 2;
  }
  return /[.!?]["']?$/.test(answer) ? answer : "";
}

async function chat(provider: Provider, messages: Array<{ role: string; content: string }>, maxTokens: number): Promise<{ text: string; finish: string }> {
  const response = await fetch(`${provider.baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${provider.apiKey}`,
      "Content-Type": "application/json",
      Accept: "application/json",
      ...provider.headers,
    },
    body: JSON.stringify({
      model: provider.model,
      temperature: 0,
      max_tokens: maxTokens,
      messages,
    }),
    signal: AbortSignal.timeout(20000),
  });
  if (!response.ok) throw new Error(`${provider.name} returned ${response.status}`);
  const body = await response.json();
  const choice = body.choices?.[0];
  return { text: sanitize(textOf(choice?.message?.content)), finish: String(choice?.finish_reason || "") };
}

function textOf(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) => (typeof part === "string" ? part : typeof part === "object" && part && "text" in part ? String(part.text ?? "") : ""))
    .join(" ");
}

function prompt(query: string, evidence: Evidence[]): string {
  const limit = numberEnv("LLM_EVIDENCE_CHARS", 700);
  const lines = forModel(evidence).map((item, index) => {
    const when = item.timestamp || "undated";
    const where = item.origin || item.entity || "source";
    return `[${index + 1}] (${when}, ${where}) ${item.title}: ${excerpt(query, item.text, limit)}`;
  });
  if (lines.length === 0) {
    return `Answer the question briefly and factually.\nQuestion: ${query}\nAnswer:`;
  }
  return [
    "Use the dated evidence to answer. Prefer evidence whose date matches the question's time intent. Cite dates when relevant. Keep the answer short.",
    "",
    "Evidence:",
    lines.join("\n"),
    "",
    `Question: ${query}`,
    "Answer:",
  ].join("\n");
}

function forModel(evidence: Evidence[]): Evidence[] {
  const perSource = numberEnv("LLM_EVIDENCE_PER_SOURCE", 3);
  const groups = new Map<string, Evidence[]>();
  for (const item of evidence) {
    const key = item.origin || item.entity || "source";
    groups.set(key, [...(groups.get(key) ?? []), item]);
  }
  return [...groups.values()].flatMap((items) => items.slice(0, perSource));
}

function excerpt(query: string, text: string, limit: number): string {
  const clean = text.replace(/\s+/g, " ").trim();
  const terms = query.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/).filter((term) => term.length > 2 && !STOP.has(term));
  const parts = clean.split(/(?<=[.!?])\s+(?=[A-Z])/).map((sentence) => sentence.trim()).filter((sentence) => sentence.length > 40);
  const ranked = parts
    .map((sentence) => ({ sentence, score: terms.reduce((hits, term) => hits + (sentence.toLowerCase().includes(term) ? 1 : 0), 0) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, 2)
    .map((item) => item.sentence);
  const chosen = parts.filter((sentence) => ranked.includes(sentence));
  const snippet = (chosen.length > 0 ? chosen : [clean]).join(" ");
  return snippet.length <= limit ? snippet : `${snippet.slice(0, limit).replace(/\s+\S*$/, "")}…`;
}

function sanitize(text: string): string {
  return text.replace(/[\u00a0\u202f\u2000-\u200b]/g, " ").replace(/【\d+】/g, "").replace(/\[\d+\]/g, "").replace(/[ \t]{2,}/g, " ").trim();
}

function numberEnv(name: string, fallback: number): number {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}
