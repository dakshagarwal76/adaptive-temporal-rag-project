"use client";

import { useState } from "react";
import { EXAMPLES, type AskResult, type SystemId } from "@/lib/engine";

const SYSTEMS: Array<{ id: SystemId; label: string }> = [
  { id: "adaptive", label: "Adaptive" },
  { id: "standard", label: "Standard RAG" },
  { id: "fixed", label: "Fixed window" },
  { id: "closed", label: "Closed book" },
];

export function AskPanel() {
  const [query, setQuery] = useState(EXAMPLES[0]);
  const [system, setSystem] = useState<SystemId>("adaptive");
  const [result, setResult] = useState<AskResult | null>(null);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);

  async function run(nextQuery = query, nextSystem = system) {
    const text = nextQuery.trim();
    if (!text) {
      setError("Enter a question.");
      return;
    }
    setPending(true);
    setError("");
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text, system: nextSystem }),
      });
      const payload = await response.json();
      if (!response.ok) {
        setError(payload.error || "The request failed.");
        setResult(null);
        return;
      }
      setResult(payload as AskResult);
    } catch {
      setError("Could not reach the server.");
      setResult(null);
    } finally {
      setPending(false);
    }
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
        <div className="systems" role="group" aria-label="System">
          {SYSTEMS.map((item) => (
            <button
              key={item.id}
              type="button"
              data-on={system === item.id ? "true" : "false"}
              onClick={() => {
                setSystem(item.id);
                if (query.trim()) void run(query, item.id);
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
      </form>

      <div className="examples">
        {EXAMPLES.map((example) => (
          <button
            key={example}
            type="button"
            onClick={() => {
              setQuery(example);
              void run(example, system);
            }}
          >
            {example}
          </button>
        ))}
      </div>

      {error ? <p className="error">{error}</p> : null}

      {result ? (
        <section className="result">
          <p className="answer">{result.answer}</p>
          <div className="meta">
            <span className="chip">{labelFor(result.system)}</span>
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
                <dd>{baselinePolicy(result.system)}</dd>
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
                    <span><strong>{index + 1}. {doc.timestamp}</strong> · {doc.title}</span>
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

function labelFor(system: SystemId) {
  return SYSTEMS.find((item) => item.id === system)?.label ?? system;
}

function baselinePolicy(system: SystemId) {
  if (system === "standard") return "Always retrieve, rank by text overlap";
  if (system === "fixed") return "Always retrieve, fixed ±2 year window or recency";
  if (system === "closed") return "Never retrieve";
  return "Adaptive";
}
