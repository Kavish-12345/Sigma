import { useEffect } from "react";
import { Logo } from "./mech.jsx";

export const PROPS = [
  ["Su", "Tensile strength", "MPa", true],
  ["Sy", "Yield strength", "MPa", true],
  ["A5", "Elongation", "%", true],
  ["E", "Young's modulus", "MPa", true],
  ["G", "Shear modulus", "MPa", false],
  ["mu", "Poisson's ratio", "", false],
  ["Ro", "Density", "kg/m³", true],
];

// Percentile of a value among all materials in the dataset (0-100).
export function percentile(summary, key, value) {
  if (!summary || value === null || value === undefined) return null;
  const vals = summary.materials.map((m) => m[key]).filter((v) => typeof v === "number");
  if (!vals.length) return null;
  const below = vals.filter((v) => v < value).length;
  return Math.round((below / vals.length) * 100);
}

function useEsc(onClose) {
  useEffect(() => {
    const k = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
}

// Manufacturer-style datasheet for one recommended material (printable).
export default function Datasheet({ r, summary, partName, onClose }) {
  useEsc(onClose);
  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal datasheet printable" onClick={(e) => e.stopPropagation()}>
        <div className="ds-head">
          <div className="brand">
            <Logo size={30} />
            <div>
              <div className="wordmark">SIGMA</div>
              <div className="muted small">Material datasheet</div>
            </div>
          </div>
          <div className="ds-actions no-print">
            <button className="ghost small" onClick={() => window.print()}>Print / Save PDF</button>
            <button className="qclose" onClick={onClose} aria-label="Close">✕</button>
          </div>
        </div>

        <div className="ds-title">
          <div>
            <h2>{r.material}</h2>
            <p className="muted">
              {r.material_group_label} · {r.standard} · {r.heat_treatment || "condition not stated"}
            </p>
          </div>
          <div className="ds-rank">
            <span>RANK</span>
            <b>#{r.rank}</b>
          </div>
        </div>

        <div className="ds-grid">
          <section>
            <h4>Mechanical & physical properties</h4>
            <table className="ds-table">
              <thead>
                <tr><th>Property</th><th>Value</th><th>Position in dataset</th></tr>
              </thead>
              <tbody>
                {PROPS.map(([k, label, unit, showPct]) => {
                  const v = r.properties[k];
                  const p = showPct ? percentile(summary, k, v) : null;
                  return (
                    <tr key={k}>
                      <td>{label}</td>
                      <td className="mono">
                        {v ?? "—"} {v != null && unit}
                        {k === "A5" && r.a5_imputed && <em className="muted"> *</em>}
                      </td>
                      <td>
                        {p !== null ? (
                          <div className="pct">
                            <div className="pct-bar"><div style={{ width: `${p}%` }} /></div>
                            <span className="mono">{k === "Ro" ? `${100 - p}% heavier` : `top ${Math.max(1, 100 - p)}%`}</span>
                          </div>
                        ) : (
                          <span className="muted">—</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            {r.a5_imputed && <p className="muted small">* Elongation missing in the dataset; filled with the family median.</p>}
          </section>

          <section>
            <h4>Suitability (ML model)</h4>
            <div className="ds-big mono">{r.suitability_pct.toFixed(0)}%</div>
            {Object.entries(r.class_suitability_pct).map(([c, p]) => (
              <div className="wbar" key={c}>
                <span>{c}</span>
                <div className="bar"><div style={{ width: `${p}%` }} /></div>
                <span className="mono">{p.toFixed(0)}%</span>
              </div>
            ))}
            <div className="kv"><span>TOPSIS score</span><b className="mono">{r.topsis_score.toFixed(3)}</b></div>
            <div className="kv"><span>Corrosion score</span><b className="mono">{r.corrosion_score}/5</b></div>
            <div className="kv"><span>Dataset label</span><b>{r.labelled_classes}</b></div>

            <h4>Cost</h4>
            <div className="kv"><span>Price</span><b className="mono">{r.price.text}</b></div>
            <div className="kv"><span>Material cost</span><b className="mono">{r.cost.text}</b></div>
            <div className="kv"><span>Vs #1</span><b>{r.cost.vs_top}</b></div>
            <p className="muted small">
              {r.price.why} Source: {r.price.source} (retrieved {r.price.date}). Indicative market range, not a quote; material cost only.
            </p>
          </section>
        </div>

        <div className="titleblock mini">
          <div className="tb-cell wide"><span>PART</span><b>{partName || "—"}</b></div>
          <div className="tb-cell"><span>MATERIAL</span><b>{r.material}</b></div>
          <div className="tb-cell"><span>DATE</span><b>{new Date().toLocaleDateString()}</b></div>
          <div className="tb-cell"><span>SOURCE</span><b>Dept. dataset · ML + TOPSIS</b></div>
        </div>
      </div>
    </div>
  );
}

// Side-by-side comparison of up to 3 materials; best value per row highlighted.
export function CompareTable({ items, summary, onClose }) {
  useEsc(onClose);
  const rows = [
    ["Suitability", (r) => r.suitability_pct, "%", "max"],
    ["TOPSIS score", (r) => r.topsis_score, "", "max"],
    ["Tensile strength Su", (r) => r.properties.Su, "MPa", "max"],
    ["Yield strength Sy", (r) => r.properties.Sy, "MPa", "max"],
    ["Elongation A5", (r) => r.properties.A5, "%", "max"],
    ["Young's modulus E", (r) => r.properties.E, "MPa", "max"],
    ["Density ρ", (r) => r.properties.Ro, "kg/m³", "min"],
    ["Corrosion score", (r) => r.corrosion_score, "/5", "max"],
    ["Cost (typical)", (r) => (r.cost.low != null ? (r.cost.low + r.cost.high) / 2 : null), "", "min"],
  ];
  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal compare" onClick={(e) => e.stopPropagation()}>
        <div className="ds-head">
          <h3>Compare materials</h3>
          <button className="qclose" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="compare-scroll">
          <table className="ds-table cmp">
            <thead>
              <tr>
                <th></th>
                {items.map((r) => (
                  <th key={r.material}>#{r.rank} {r.material}<div className="muted small">{r.material_group_label}</div></th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map(([label, get, unit, best]) => {
                const vals = items.map(get);
                const nums = vals.filter((v) => typeof v === "number");
                const target = nums.length ? (best === "max" ? Math.max(...nums) : Math.min(...nums)) : null;
                return (
                  <tr key={label}>
                    <td>{label}</td>
                    {vals.map((v, i) => (
                      <td key={i} className={`mono ${v === target && nums.length > 1 ? "best" : ""}`}>
                        {v == null ? "—" : `${Number.isInteger(v) ? v : v.toFixed(label.startsWith("Cost") ? 2 : 3).replace(/\.?0+$/, "")} ${unit}`}
                      </td>
                    ))}
                  </tr>
                );
              })}
              <tr>
                <td>Cost range</td>
                {items.map((r) => <td key={r.material} className="mono">{r.cost.text}</td>)}
              </tr>
            </tbody>
          </table>
        </div>
        <p className="muted small">Highlighted = best value in the row. Cost is material cost only, from cited market ranges.</p>
      </div>
    </div>
  );
}
