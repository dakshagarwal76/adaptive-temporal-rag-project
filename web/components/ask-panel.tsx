"use client";

import { useEffect, useRef, useState } from "react";
import { EXAMPLES, type AskResult, type SourceId, type SystemId } from "@/lib/engine";

const SYSTEMS: Array<{ id: SystemId; label: string }> = [
  { id: "adaptive", label: "Adaptive" },
  { id: "standard", label: "Standard RAG" },
  { id: "fixed", label: "Fixed window" },
  { id: "closed", label: "Closed book" },
];

const SOURCES: Array<{ id: SourceId; label: string }> = [
  { id: "corpus", label: "corpus.jsonl" },
  { id: "wikipedia", label: "Wikipedia" },
  { id: "wikidata", label: "Wikidata" },
  { id: "gdelt", label: "GDELT" },
];

export function AskPanel() {
  const [query, setQuery] = useState(EXAMPLES[0]);
  const [system, setSystem] = useState<SystemId>("adaptive");
  const [sources, setSources] = useState<SourceId[]>(["corpus"]);
  const [model, setModel] = useState("auto");
  const [models, setModels] = useState<Array<{ id: string; provider: string; model: string }>>([]);
  const [result, setResult] = useState<AskResult | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const requestId = useRef(0);

  useEffect(() => {
    let active = true;
    fetch("/api/models")
      .then((response) => response.json())
      .then((payload) => {
        if (active && Array.isArray(payload.models)) setModels(payload.models);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, []);

  function toggleSource(id: SourceId) {
    const next = sources.includes(id) ? sources.filter((item) => item !== id) : [...sources, id];
    if (next.length === 0) return;
    const ordered = SOURCES.map((item) => item.id).filter((item) => next.includes(item));
    setSources(ordered);
    if (query.trim()) void run(query, system, ordered, model);
  }

  async function run(nextQuery = query, nextSystem = system, nextSources = sources, nextModel = model) {
    const text = nextQuery.trim();
    if (!text) {
      setError("Enter a question.");
      return;
    }
    const id = ++requestId.current;
    setPending(true);
    setError("");
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, system: nextSystem, sources: nextSources, model: nextModel }),
      });
      const payload = await response.json();
      if (id !== requestId.current) return;
      if (!response.ok) {
        setError(payload.error || "The request failed.");
        setResult(null);
        return;
      }
      setResult(payload as AskResult);
    } catch {
      if (id !== requestId.current) return;
      setError("Could not reach the server.");
      setResult(null);
    } finally {
      if (id === requestId.current) setPending(false);
    }
  }

  function chooseModel(next: string) {
    setModel(next);
    if (query.trim()) void run(query, system, sources, next);
  }

  return (
    <>
      <form
        className="panel"
        onSubmit={(event) => {
          event.preventDefault();
          void run();
        }}
      >
        <div className="ask-row">
          <input
            aria-label="Question"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask a time-sensitive question"
          />
          <button type="submit" disabled={pending}>{pending ? "Running" : "Ask"}</button>
        </div>
        <div className="choice">
          <span className="choice-label" id="sources-label">Sources</span>
          <div className="systems" role="group" aria-labelledby="sources-label">
            {SOURCES.map((item) => (
              <button
                key={item.id}
                type="button"
                aria-pressed={sources.includes(item.id)}
                data-on={sources.includes(item.id) ? "true" : "false"}
                onClick={() => toggleSource(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <div className="choice">
          <span className="choice-label" id="type-label">Type</span>
          <div className="systems" role="group" aria-labelledby="type-label">
            {SYSTEMS.map((item) => (
              <button
                key={item.id}
                type="button"
                data-on={system === item.id ? "true" : "false"}
                onClick={() => {
                setSystem(item.id);
                if (query.trim()) void run(query, item.id, sources, model);
                }}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
        <div className="choice">
          <span className="choice-label" id="model-label">Model</span>
          <div className="systems" role="group" aria-labelledby="model-label">
            <button type="button" data-on={model === "auto" ? "true" : "false"} onClick={() => chooseModel("auto")}>Auto</button>
            {models.map((item) => (
              <button key={item.id} type="button" data-on={model === item.id ? "true" : "false"} onClick={() => chooseModel(item.id)}>
                {item.provider} · {item.model}
              </button>
            ))}
          </div>
        </div>
      </form>

      <div className="examples">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => {
              setQuery(example);
              void run(example, system, sources, model);
            }}
          >
            {example}
          </button>
        ))}
      </div>

      {error ? <p className="error">{error}</p> : null}

      {result ? (
        <section className="result">
          {result.warnings?.map((warning) => (
            <p className="error" key={warning}>{warning}</p>
          ))}
          <p className="answer">{result.answer}</p>
          <div className="meta">
            {(result.sources ?? [result.source]).map((item) => (
              <span className="chip" key={item}>{sourceLabel(item)}</span>
            ))}
            <span className="chip">{labelFor(result.system)}</span>
            <span className="chip">{backendLabel(result.backend)}</span>
            <span className="chip">{result.didRetrieve ? "Retrieved" : "No retrieval"}</span>
            {result.analyzer ? <span className="chip">{result.analyzer.temporalClass}</span> : null}
            <span className="chip">{result.latencyMs} ms</span>
          </div>

          <dl className="facts">
            {result.analyzer ? (
              <>
                <dt>Policy</dt>
                <dd>{result.analyzer.policy.replaceAll("_", " ")}</dd>
                <dt>Router confidence</dt>
                <dd>{result.analyzer.confidence.toFixed(2)}</dd>
              </>
            ) : (
              <>
                <dt>Policy</dt>
                <dd>{baselinePolicy(result.system, result.sources ?? [result.source])}</dd>
              </>
            )}
            {result.window ? (
              <>
                <dt>Date window</dt>
                <dd>
                  {result.window.targetYear ?? "—"} · ±{result.window.finalYears}y · {result.window.reason}
                </dd>
              </>
            ) : null}
            {result.sufficiency ? (
              <>
                <dt>Sufficiency</dt>
                <dd>
                  {result.sufficiency.score.toFixed(2)} · {result.sufficiency.sufficient ? "enough evidence" : "thin evidence"}
                </dd>
              </>
            ) : null}
          </dl>

          <div className="evidence">
            <h2>Evidence</h2>
            {result.evidence.length === 0 ? (
              <p className="empty">Nothing was retrieved for this question.</p>
            ) : (
              result.evidence.map((doc, index) => (
                <article className="card" key={doc.docId}>
                  <header>
                    <span>
                      {(result.sources?.length ?? 0) > 1 ? `${sourceLabel(doc.origin ?? result.source)} · ` : null}
                      <strong>{index + 1}. {doc.timestamp || sourceLabel(doc.origin ?? result.source)}</strong>
                      {" · "}
                      {doc.url ? <a href={doc.url} target="_blank" rel="noreferrer">{doc.title}</a> : doc.title}
                    </span>
                    <span>{doc.score.toFixed(2)}</span>
                  </header>
                  <p>{doc.text}</p>
                </article>
              ))
            )}
          </div>
        </section>
      ) : null}
    </>
  );
}

function backendLabel(backend: string) {
  const match = backend.match(/^api:([^:]+):(.+)$/);
  return match ? `${match[1]} · ${match[2]}` : backend;
}

function labelFor(system: SystemId) {
  return SYSTEMS.find((item) => item.id === system)?.label ?? system;
}

function sourceLabel(source: AskResult["source"]) {
  return SOURCES.find((item) => item.id === source)?.label ?? source;
}

function baselinePolicy(system: SystemId, sources: SourceId[]) {
  if (sources.length > 1) return sources.map((source) => `${sourceLabel(source)}: ${policyFor(system, source)}`).join(" · ");
  return policyFor(system, sources[0] ?? "corpus");
}

function policyFor(system: SystemId, source: SourceId) {
  if (source === "wikipedia" && system !== "closed") return "Wikipedia search";
  if (source === "wikidata" && system !== "closed") return "Wikidata items";
  if (source === "gdelt" && system !== "closed") return "GDELT news, last 3 months";
  if (system === "standard") return "Always retrieve, rank by text overlap";
  if (system === "fixed") return "Always retrieve, fixed ±2 year window or recency";
  if (system === "closed") return "Never retrieve";
  return "Retrieve the main passages for that question";
}
