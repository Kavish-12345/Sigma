import { useState } from "react";

// Material selection in the design process (Ashby): translation -> screening -> ranking.
const PROP_NAME = { Sy: "Yield strength", Su: "Tensile strength", E: "Young's modulus" };
const WEIGHT_NAME = { suitability: "Suitability", index: "Strength index", cost: "Cost", density: "Weight", corrosion: "Corrosion" };

function Summary({ design, result, notes }) {
  const t = design?.translation;
  return (
    <>
      {t && (
        <dl className="dp-facts">
          <dt>Part</dt>
          <dd>{t.function}</dd>
          <dt>Inputs</dt>
          <dd>{t.constraints.join(" · ")}</dd>
          <dt>Goal</dt>
          <dd>{t.objective}</dd>
        </dl>
      )}
      {design && (
        <>
          <h4>Required</h4>
          <div className="dp-req">
            {Object.entries(design.requirements).map(([k, v]) => (
              <span key={k} className="chip on">
                {k === "A5" ? `Elongation ≥ ${v}%` : `${PROP_NAME[k] || k} ≥ ${Number(v).toLocaleString()} MPa`}
              </span>
            ))}
            {design.min_corrosion_score && <span className="chip on">Corrosion score ≥ {design.min_corrosion_score}/5</span>}
          </div>
          <p className="muted small">
            {result.n_candidates} of {design.n_screened_out + result.n_candidates} materials pass these limits and your filters.
          </p>
        </>
      )}
      {notes.length > 0 && (
        <>
          <h4>Check for this part</h4>
          <ul className="dp-notes">
            {notes.map((n) => (
              <li key={n}>{n}</li>
            ))}
          </ul>
        </>
      )}
    </>
  );
}

function Calculation({ design }) {
  return (
    <ol className="dp-calc mono">
      {design.steps.map((s, i) => (
        <li key={i}>{s}</li>
      ))}
    </ol>
  );
}

function Ranking({ design, result, onChart }) {
  const weights = Object.entries(result.weights_normalised).filter(([, w]) => w > 0).sort((a, b) => b[1] - a[1]);
  return (
    <>
      <p className="muted">Materials that pass are scored on these factors with TOPSIS. Usual engineering practice for this part then adjusts the order.</p>
      <div className="dp-weights">
        {weights.map(([k, w]) => (
          <div key={k} className="dp-weight">
            <span>{WEIGHT_NAME[k] || k}</span>
            <div className="bar"><div style={{ width: `${w * 100}%` }} /></div>
            <strong>{(w * 100).toFixed(0)}%</strong>
          </div>
        ))}
      </div>
      <p className="muted small">
        Strength index M = <span className="mono">{design.index.text}</span>. Suitability is 80% ML model and 20% engineering properties.
      </p>
      {onChart && (
        <button className="link" onClick={onChart}>
          Show the Ashby chart →
        </button>
      )}
    </>
  );
}

export default function DesignPanel({ design, result, notes = [], onChart }) {
  const [tab, setTab] = useState("Summary");
  const tabs = design ? ["Summary", "Calculation", "Ranking"] : ["Summary"];
  return (
    <section className="design-panel">
      {tabs.length > 1 && (
        <div className="dp-tabs" role="tablist">
          {tabs.map((t) => (
            <button key={t} role="tab" aria-selected={t === tab} className={t === tab ? "on" : ""} onClick={() => setTab(t)}>
              {t}
            </button>
          ))}
        </div>
      )}
      {tab === "Summary" && <Summary design={design} result={result} notes={notes} />}
      {tab === "Calculation" && <Calculation design={design} />}
      {tab === "Ranking" && <Ranking design={design} result={result} onChart={onChart} />}
    </section>
  );
}
