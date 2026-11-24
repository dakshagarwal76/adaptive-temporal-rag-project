const ROWS = [
  { name: "Closed book", em: "0.130", f1: "0.167", contains: "0.130", p5: "0.000", temporal: "0.247", retrieval: "0%" },
  { name: "Standard RAG", em: "0.649", f1: "0.676", contains: "0.727", p5: "0.411", temporal: "0.662", retrieval: "100%" },
  { name: "Fixed window", em: "0.935", f1: "0.951", contains: "0.987", p5: "0.608", temporal: "1.000", retrieval: "100%" },
  { name: "Adaptive", em: "0.948", f1: "0.960", contains: "0.987", p5: "0.668", temporal: "1.000", retrieval: "87%", ours: true },
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
        Numbers are the saved synthetic run. The live page answers with the same corpus and the temporal policy: lexical retrieval, an adaptive date window, and an extractive reader. The GPU embedding and router models stay with the research artifacts.
      </p>
    </main>
  );
}
