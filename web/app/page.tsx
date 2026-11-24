import { AskPanel } from "@/components/ask-panel";

export default function HomePage() {
  return (
    <main>
      <div className="lede">
        <h1>When and what to retrieve.</h1>
        <p>
          Select corpus.jsonl, Wikipedia, Wikidata, GDELT, or any combination. The configured model writes the answer from the evidence that was retrieved. Adaptive retrieval skips trivia, keeps a window around a past year, prefers the latest passage for a current office, and returns only that leading evidence.
        </p>
      </div>
      <AskPanel />
    </main>
  );
}
