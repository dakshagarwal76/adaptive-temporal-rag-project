const ROWS = [
  { name: "Closed book", em: "0.130", f1: "0.167", contains: "0.130", p5: "0.000", temporal: "0.247", retrieval: "0%" },
  { name: "Standard RAG", em: "0.649", f1: "0.676", contains: "0.727", p5: "0.411", temporal: "0.662", retrieval: "100%" },
  { name: "Fixed window", em: "0.935", f1: "0.951", contains: "0.987", p5: "0.608", temporal: "1.000", retrieval: "100%" },
  { name: "Adaptive", em: "0.948", f1: "0.960", contains: "0.987", p5: "0.668", temporal: "1.000", retrieval: "87%", ours: true },
];

const ROUTER = [
  { name: "r4 base", model: "DeBERTa-v3-base", em: "0.948", heldout: "0.762", timeqa: "0.296", ece: "0.0559", probes: "29/36", challenge: "65/72", ours: false },
  { name: "r4 large", model: "DeBERTa-v3-large", em: "0.948", heldout: "0.762", timeqa: "0.296", ece: "0.0478", probes: "29/36", challenge: "66/72", ours: true },
];

export default function ResultsPage() {
  return (
    <main>
      <div className="lede">
        <h1>Saved evaluation.</h1>
        <p>
          Seventy-seven templated questions, same split for every system. Exact match, token F1, and containment use the project’s answer normalisation.
        </p>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>System</th>
              <th>EM</th>
              <th>Token F1</th>
              <th>Contains</th>
              <th>P@5</th>
              <th>Temporal</th>
              <th>Retrieved</th>
            </tr>
          </thead>
          <tbody>
            {ROWS.map((row) => (
              <tr key={row.name} data-ours={row.ours ? "true" : "false"}>
                <td>{row.name}</td>
                <td>{row.em}</td>
                <td>{row.f1}</td>
                <td>{row.contains}</td>
                <td>{row.p5}</td>
                <td>{row.temporal}</td>
                <td>{row.retrieval}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="note">
        Checkpoint B, 77 templated questions. The Ask page uses this rule policy: lexical retrieval, an adaptive date window, and an extractive reader.
      </p>

      <section className="section">
        <h2>Learned router, 24 Nov 2026.</h2>
        <p>
          DeBERTa trained on 80,145 questions. On the frozen answer tests both checkpoints tie the Adaptive row above: no discordant answers, McNemar p = 1. The large model is the better calibrated of the two.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Router</th>
                <th>Model</th>
                <th>EM</th>
                <th>Held-out EM</th>
                <th>TimeQA contains</th>
                <th>ECE</th>
                <th>36 probes</th>
                <th>72 challenge</th>
              </tr>
            </thead>
            <tbody>
              {ROUTER.map((row) => (
                <tr key={row.name} data-ours={row.ours ? "true" : "false"}>
                  <td>{row.name}</td>
                  <td>{row.model}</td>
                  <td>{row.em}</td>
                  <td>{row.heldout}</td>
                  <td>{row.timeqa}</td>
                  <td>{row.ece}</td>
                  <td>{row.probes}</td>
                  <td>{row.challenge}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">
          Base trained 3 epochs in 2 min 51 s. Large trained 2 epochs in 6 min 47 s. Probe and challenge cells count questions where both the time label and the retrieval decision were correct. Those weights are not loaded by this site.
        </p>
      </section>
    </main>
  );
}
