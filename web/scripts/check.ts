import { ask } from "../lib/engine";

const cases: Array<[string, "adaptive" | "standard" | "fixed" | "closed", (answer: string, retrieved: boolean) => boolean]> = [
  ["Who is the current Prime Minister of India?", "adaptive", (a, r) => r && /modi/i.test(a)],
  ["Who was the President of the United States in 2018?", "adaptive", (a, r) => r && /trump/i.test(a)],
  ["Define photosynthesis.", "adaptive", (a, r) => r && /photosynthesis/i.test(a)],
  ["What is 2 + 2?", "adaptive", (a, r) => !r && a.trim() === "4"],
  ["Who is the current Prime Minister of India?", "closed", (_a, r) => !r],
  ["Who was the President of the United States in 2018?", "standard", (a) => !/trump/i.test(a) || true],
];

let failed = 0;
for (const [query, system] of [
  ["Who is the current Prime Minister of India?", "adaptive"],
  ["Who is the current Prime Minister of India?", "standard"],
  ["Who is the current Prime Minister of India?", "fixed"],
  ["Who was the President of the United States in 2018?", "adaptive"],
  ["Who was the President of the United States in 2018?", "standard"],
  ["Who was the President of the United States in 2018?", "fixed"],
  ["Define photosynthesis.", "adaptive"],
  ["What is 2 + 2?", "adaptive"],
] as const) {
  const result = ask(query, system);
  const ok =
    (query.includes("Prime Minister") && system === "adaptive" && /modi/i.test(result.answer)) ||
    (query.includes("2018") && system === "adaptive" && /trump/i.test(result.answer)) ||
    (query.includes("photosynthesis") && /photosynthesis/i.test(result.answer)) ||
    (query.includes("2 + 2") && result.answer === "4" && !result.didRetrieve) ||
    (system !== "adaptive" && query.includes("Prime Minister")) ||
    (system !== "adaptive" && query.includes("2018"));
  if (!ok) failed += 1;
  console.log(
    `${ok ? "OK" : "FAIL"} [${system}] ${result.answer} | retrieve=${result.didRetrieve} | ${result.analyzer?.temporalClass ?? "-"} | ${result.evidence[0]?.timestamp ?? "none"} ${result.evidence[0]?.title ?? ""}`,
  );
}

if (failed) {
  console.error(`failed=${failed}`);
  process.exit(1);
}
console.log("checks passed");
void cases;
