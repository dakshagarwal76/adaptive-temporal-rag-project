import { AskPanel } from "@/components/ask-panel";

export default function HomePage() {
  return (
    <main>
      <div className="lede">
        <h1>When and what to retrieve.</h1>
        <p>
          Ask a question against corpus.jsonl, or switch the source to Wikipedia. Adaptive retrieval skips trivia, keeps a window around a past year, and prefers the latest passage for a current office.
        </p>
      </div>
      <AskPanel />
    </main>
  );
}
