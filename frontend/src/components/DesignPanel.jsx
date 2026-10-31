// Material selection in the design process (Ashby): translation -> screening -> ranking -> supporting info.
const PROP_NAME = { Sy: "Yield strength Sy", Su: "Tensile strength Su", E: "Young's modulus E" };

export default function DesignPanel({ design, result, onChart }) {
  if (!design) return null;
  const t = design.translation;
  const top = result.results[0];
  return (
    <section className="design-panel">
      <div className="dp-head">
        <span className="mono small">MATERIAL SELECTION IN THE DESIGN PROCESS</span>
        <h3>{design.function_label}</h3>
      </div>

      <div className="dp-steps">
        <div className="dp-step">
          <div className="dp-num">1</div>
          <div>
            <h4>Translation</h4>
            <dl>
              <dt>Function</dt>
              <dd>{t.function}</dd>
              <dt>Constraints</dt>
              <dd>{t.constraints.join(" · ")}</dd>
              <dt>Objective</dt>
              <dd>{t.objective}</dd>
              <dt>Free variable</dt>
              <dd>{t.free_variable}</dd>
            </dl>
          </div>
        </div>

        <div className="dp-step">
          <div className="dp-num">2</div>
          <div>
            <h4>Design calculation → screening</h4>
            <ol className="dp-calc mono">
              {design.steps.map((s, i) => (
                <li key={i}>{s}</li>
              ))}
            </ol>
            <div className="dp-req">
              {Object.entries(design.requirements).map(([k, v]) => (
                <span key={k} className="chip on">
                  {PROP_NAME[k] || k} ≥ {Number(v).toLocaleString()} MPa
                </span>
              ))}
              {design.min_corrosion_score && <span className="chip on">Corrosion score ≥ {design.min_corrosion_score}/5 (environment)</span>}
            </div>
            <p className="muted small">
              {design.n_screened_out} of {design.n_screened_out + result.n_candidates} materials removed by these
              limits and your other filters; {result.n_candidates} pass.
              {design.targets_from_function && ` Application classes taken from the function: ${Object.keys(result.targets).join(", ")}.`}
            </p>
          </div>
        </div>

        <div className="dp-step">
          <div className="dp-num">3</div>
          <div>
            <h4>Ranking</h4>
            <p>
              Material index <strong className="mono">M = {design.index.text}</strong> (larger is better), combined with ML
              suitability, cost, density and corrosion by TOPSIS.
            </p>
            <p className="muted small">
              Weights:{" "}
              {Object.entries(result.weights_normalised)
                .filter(([, w]) => w > 0)
                .map(([k, w]) => `${k} ${(w * 100).toFixed(0)}%`)
                .join(" · ")}
              {design.index.objective === "min_cost" && " · the chart line shows the mass index " + design.index.mass_text}
            </p>
            {onChart && (
              <button className="link" onClick={onChart}>
                Show the selection line on the Ashby chart →
              </button>
            )}
          </div>
        </div>

        <div className="dp-step">
          <div className="dp-num">4</div>
          <div>
            <h4>Supporting information → final choice</h4>
            {top?.design_check && (
              <p>
                <strong>{top.display_name}</strong>:{" "}
                {top.design_check
                  .map((c) => (c.fos != null ? `${c.name} FoS ${c.fos}` : `${c.name} ${c.margin}× required`))
                  .join(" · ")}
              </p>
            )}
            <p className="muted small">Each card below shows its design check, datasheet, price and heat treatment.</p>
          </div>
        </div>
      </div>
    </section>
  );
}
