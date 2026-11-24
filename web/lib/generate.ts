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
  error?: string;
};

const SYSTEM = "You are a concise factual assistant. The evidence is one combined collection, even when the passages come from different sources. Write one answer in one or two complete sentences, grounded in that evidence, and include the passage date. If a passage names the person or fact and its date is near the year in the question, answer from that passage. Do not say the evidence is missing in that case.";
const SYSTEM_TIMELESS = "You are a concise factual assistant. This question does not depend on a date. Answer it in one or two complete sentences. Use a passage only when it answers the question. If every passage is about something else, ignore the passages and answer the question yourself.";
const SYSTEM_OPEN = "You are a concise factual assistant. No passages were retrieved. Answer the question in one or two complete sentences.";
const downHosts = new Set<string>();
const STOP = new Set(["who", "what", "which", "when", "where", "whom", "whose", "the", "a", "an", "of", "is", "are", "was", "were", "did", "does", "how", "for", "and", "current", "currently"]);

export function fallbackAnswer(query: string, evidence: Evidence[]): string {
  let best = "";
  let bestScore = 0;
  for (const item of evidence) {
    const sentence = excerpt(query, item.text, 400);
    const score = overlap(query, sentence);
    if (score > bestScore) {
      best = sentence;
      bestScore = score;
    }
  }
  return best;
}

function overlap(query: string, text: string): number {
  const terms = query.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/).filter((term) => term.length > 2 && !STOP.has(term));
  const haystack = text.toLowerCase();
  return terms.reduce((hits, term) => hits + (haystack.includes(term) ? 1 : 0), 0);
}

export type ModelChoice = { id: string; provider: string; model: string };

export function configuredModels(): ModelChoice[] {
  return loadProviders().map((provider) => ({
    id: `${provider.name}:${provider.model}`,
    provider: provider.name,
    model: provider.model,
  }));
}

export async function generateAnswer(query: string, evidence: Evidence[], choice?: string, timeless = false): Promise<Generation | null> {
  const mode = env("GENERATOR_BACKEND").trim().toLowerCase() || "api";
  if (mode === "extractive" || mode === "hf") return null;
  let providers = loadProviders().filter((provider) => !downHosts.has(provider.baseUrl));
  if (choice && choice !== "auto") {
    const preferred = loadProviders().filter((provider) => `${provider.name}:${provider.model}` === choice);
    if (preferred.length === 0) {
      return { answer: "", backend: "extractive", error: "Model failed. That model is not configured." };
    }
    const others = loadProviders().filter((provider) => !preferred.some((item) => item.name === provider.name && item.model === provider.model) && !downHosts.has(provider.baseUrl));
    providers = [...preferred.filter((provider) => !downHosts.has(provider.baseUrl)), ...others];
    if (providers.length === 0) providers = preferred;
  }
  if (providers.length === 0) {
    return { answer: "", backend: "extractive", error: "Model failed. No provider is configured. A provider needs an API key, base URL, and model." };
  }
  const lead = choice && choice !== "auto"
    ? providers.filter((provider) => `${provider.name}:${provider.model}` === choice)
    : providers.slice(0, 1);
  const errors: string[] = [];
  let allowRest = evidence.length === 0;
  for (const provider of providers) {
    const isLead = lead.some((item) => item.name === provider.name && item.model === provider.model);
    if (downHosts.has(provider.baseUrl)) continue;
    if (!isLead && !allowRest) break;
    try {
      let result = await complete(provider, query, evidence, timeless);
      if (!result.text && evidence.length > 0) result = await complete(provider, query, [], true);
      if (result.text) return { answer: result.text, backend: `api:${provider.name}:${provider.model}` };
      errors.push(`${provider.name} (${provider.model}) returned an empty answer.`);
      if (timeless && isLead) break;
      if (isLead) allowRest = evidence.length === 0;
    } catch (error) {
      const detail = messageOf(error);
      errors.push(`${provider.name} (${provider.model}): ${detail}`);
      if (gatewayFailure(detail)) downHosts.add(provider.baseUrl);
      if (timeless && isLead) break;
      if (isLead) allowRest = evidence.length === 0 || gatewayFailure(detail);
    }
  }
  return { answer: "", backend: "extractive", error: errors.length > 0 ? `Model failed. ${errors.join(" ")}` : "Model failed." };
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
    const models = modelList(env(`${prefix}_MODELS`), env(`${prefix}_MODEL`));
    if (!apiKey || !baseUrl || models.length === 0) continue;
    const headers: Record<string, string> = {};
    const site = env(`${prefix}_SITE_URL`);
    const app = env(`${prefix}_APP_NAME`);
    if (site) headers["HTTP-Referer"] = site;
    if (app) headers["X-Title"] = app;
    for (const model of models) {
      providers.push({ name, apiKey, baseUrl: baseUrl.replace(/\/$/, ""), model, headers });
    }
  }
  return providers;
}

function list(value: string): string[] {
  return value
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
}

function modelList(models: string, model: string): string[] {
  const many = models.split(",").map((item) => item.trim()).filter(Boolean);
  if (many.length > 0) return many;
  return model.trim() ? [model.trim()] : [];
}

function env(name: string): string {
  return process.env[name] || "";
}

async function complete(provider: Provider, query: string, evidence: Evidence[], timeless = false): Promise<{ text: string; finish: string }> {
  const messages = [
    { role: "system", content: instructions(evidence, timeless) },
    { role: "user", content: prompt(query, evidence) },
  ];
  let budget = numberEnv("LLM_MAX_TOKENS", 512);
  let result = { text: "", finish: "" };
  for (let attempt = 0; attempt < 2; attempt += 1) {
    result = await chat(provider, messages, budget);
    if (result.text && result.finish !== "length") {
      return /[.!?]["']?$/.test(result.text) ? result : { ...result, text: `${result.text}.` };
    }
    budget *= 2;
  }
  if (!result.text) return result;
  return /[.!?]["']?$/.test(result.text) ? result : { ...result, text: `${result.text}.` };
}

async function chat(provider: Provider, messages: Array<{ role: string; content: string }>, maxTokens: number): Promise<{ text: string; finish: string }> {
  let lastStatus = 0;
  let lastDetail = "";
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
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
          ...(provider.name === "groq" ? { reasoning_effort: "low" } : {}),
          messages,
        }),
        signal: AbortSignal.timeout(12000),
      });
      if (response.ok) {
        const body = await response.json();
        const choice = body.choices?.[0];
        return { text: sanitize(messageText(choice?.message)), finish: String(choice?.finish_reason || "") };
      }
      lastStatus = response.status;
      lastDetail = await errorDetail(response);
      if (!gatewayStatus(response.status) || attempt === 1) break;
      await wait(400);
    } catch (error) {
      if (attempt === 1 || !transient(error)) throw error;
      await wait(400);
    }
  }
  throw new Error(lastDetail ? `returned ${lastStatus}: ${lastDetail}` : `returned ${lastStatus || 502}`);
}

function gatewayStatus(status: number): boolean {
  return status === 502 || status === 503 || status === 504;
}

function gatewayFailure(detail: string): boolean {
  return /returned 50[234]\b|fetch failed|network|timed out/i.test(detail);
}

function transient(error: unknown): boolean {
  return error instanceof Error && (error.name === "TypeError" || /fetch failed|network|ECONNRESET|ETIMEDOUT/i.test(error.message));
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function instructions(evidence: Evidence[], timeless: boolean): string {
  if (evidence.length === 0) return SYSTEM_OPEN;
  return timeless ? SYSTEM_TIMELESS : SYSTEM;
}

function messageText(message: { content?: unknown; reasoning?: unknown } | undefined): string {
  const content = textOf(message?.content);
  if (content) return content;
  const reasoning = textOf(message?.reasoning);
  if (!reasoning) return "";
  const lines = reasoning.split(/\n+/).map((line) => line.trim()).filter(Boolean);
  const last = lines[lines.length - 1] || reasoning.trim();
  return last.length <= 500 ? last : last.slice(-500).replace(/^\S*\s/, "");
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
    "Use the dated evidence to answer. Prefer a passage from the asked year. If none is, use the closest passage and state its date. Keep the answer short.",
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

async function errorDetail(response: Response): Promise<string> {
  const text = await response.text();
  if (/<!doctype html/i.test(text) || /<html/i.test(text)) return gatewayStatus(response.status) ? "bad gateway" : "the host returned a web page";
  try {
    const body = JSON.parse(text) as { error?: { message?: unknown } | string; message?: unknown; title?: unknown };
    const nested = body?.error;
    const message = typeof nested === "string" ? nested : nested?.message ?? body?.message;
    if (typeof message === "string" && message.trim()) return scrub(message.trim()).slice(0, 180);
    if (typeof body.title === "string" && body.title.trim()) return body.title.replace(/^Error\s+\d+:\s*/i, "").trim();
  } catch {
    // The provider returned plain text.
  }
  const plain = text.replace(/\s+/g, " ").trim();
  return plain ? scrub(plain).slice(0, 180) : "";
}

function messageOf(error: unknown): string {
  if (error instanceof Error && (error.name === "TimeoutError" || error.name === "AbortError")) return "the request timed out.";
  const raw = error instanceof Error ? error.message : "the request failed.";
  return scrub(raw);
}

function scrub(text: string): string {
  return text
    .replace(/Bearer\s+\S+/gi, "Bearer [redacted]")
    .replace(/\b(?:gsk_|sk-|sk-or-|grok_)[A-Za-z0-9_-]+/g, "[redacted]");
}

function sanitize(text: string): string {
  return text.replace(/[\u00a0\u202f\u2000-\u200b]/g, " ").replace(/【\d+】/g, "").replace(/\[\d+\]/g, "").replace(/[ \t]{2,}/g, " ").trim();
}

function numberEnv(name: string, fallback: number): number {
  const value = Number(process.env[name]);
  return Number.isFinite(value) && value > 0 ? value : fallback;
}
