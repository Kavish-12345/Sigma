import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

const PROPS = [
  ["Su", "Tensile strength", "MPa"],
  ["Sy", "Yield strength", "MPa"],
  ["A5", "Elongation", "%"],
  ["E", "Young's modulus", "MPa"],
  ["Ro", "Density", "kg/m³"],
  ["mu", "Poisson's ratio", ""],
];

const BASIS = { market: "Market", listing: "Listing", custom: "Your price", not_available: "No price" };

export default function ResultCard({ r, onOverride, onDatasheet, onCompare, compared, compareFull }) {
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [price, setPrice] = useState("");

  const applyPrice = () => {
    if (price) onOverride(r.material_group, price);
    setEditing(false);
    setPrice("");
  };

  return (
    <article className={`card ${r.rank === 1 ? "best" : ""}`}>
      <div className="card-top">
        <div className="rank mono">#{String(r.rank).padStart(2, "0")}</div>
        <div className="title">
          <h3>
            {r.display_name}
            {r.rank === 1 && <span className="badge best inline">Best overall</span>}
          </h3>
          <p className="muted">
            {r.material_group_label} · {r.standard}
          </p>
        </div>
        <label className={`cmp-check ${compared ? "on" : ""}`} title={compareFull && !compared ? "Up to 3" : "Add to comparison"}>
          <input type="checkbox" checked={compared} disabled={compareFull && !compared} onChange={onCompare} />
          Compare
        </label>
      </div>

      <div className="metric">
        <span>Suitability</span>
        <div className="bar big">
          <div style={{ width: `${r.suitability_pct}%` }} />
        </div>
        <strong>{r.suitability_pct.toFixed(0)}%</strong>
      </div>

      {r.design_check && (
        <div className="dcheck">
          <span className="muted small">Design check</span>
          {r.design_check.map((c) => (
            <div key={c.name} className={`dcheck-row ${c.ok ? "ok" : "bad"}`}>
              <span>{c.ok ? "✔" : "✖"} {c.name}</span>
              <span className="mono">
                {c.prop} {c.actual.toLocaleString()} vs {c.required.toLocaleString()} MPa
                {c.fos != null ? ` · FoS ${c.fos}` : ` · ${c.margin}×`}
              </span>
            </div>
          ))}
          <div className="dcheck-row">
            <span>Material index</span>
            <span className="mono">{r.index_rel_pct.toFixed(0)}% of best</span>
          </div>
        </div>
      )}

      <div className="cost">
        <div>
          <span className="muted small">Estimated material cost</span>
          <div className={`cost-value ${r.price.available ? "" : "na"}`}>{r.cost.text}</div>
          <div className="small">{r.cost.vs_top}</div>
        </div>
        <span className={`badge basis ${r.price.basis}`}>{BASIS[r.price.basis] || r.price.basis}</span>
      </div>

      <div className="row wrap card-price">
        {!editing ? (
          <button className="link" onClick={() => setEditing(true)}>
            {r.price.custom ? "Change your price" : "Enter your price"}
          </button>
        ) : (
          <div className="price-edit">
            <span>Rs/kg for {r.material_group_label}:</span>
            <input type="number" min="0" value={price} onChange={(e) => setPrice(e.target.value)} autoFocus
              onKeyDown={(e) => { if (e.key === "Enter") applyPrice(); if (e.key === "Escape") { setEditing(false); setPrice(""); } }} />
            <button className="primary small" onClick={applyPrice}>
              Apply
            </button>
            <button className="ghost small" onClick={() => { setEditing(false); setPrice(""); }}>
              Cancel
            </button>
            {r.price.custom && (
              <button className="ghost small" onClick={() => { onOverride(r.material_group, null); setEditing(false); }}>
                Reset
              </button>
            )}
          </div>
        )}
      </div>
      <div className="card-actions">
        <button className="link" onClick={() => setOpen(true)}>
          Why this material?
        </button>
        <button className="link" onClick={onDatasheet}>Datasheet</button>
      </div>

      {open && (
        <WhyModal r={r} onClose={() => setOpen(false)}>
          {r.design_check && (
            <>
              <h4>Design check (actual factor of safety)</h4>
              <ul>
                {r.design_check.map((c) => (
                  <li key={c.name}>
                    {c.name}: needs {c.prop} ≥ {c.required} MPa, this material has {c.actual} MPa
                    {c.fos != null ? ` → actual FoS ${c.fos}` : ` → ${c.margin}× the required value`}
                  </li>
                ))}
              </ul>
            </>
          )}
          <h4>Model suitability per target class</h4>
          {Object.entries(r.class_suitability_pct).map(([c, p]) => (
            <div className="wbar" key={c}>
              <span>{c}</span>
              <div className="bar">
                <div style={{ width: `${p}%` }} />
              </div>
              <span>{p.toFixed(0)}%</span>
            </div>
          ))}

          <h4>Price</h4>
          <p>
            <strong>{r.price.text}</strong> — {r.price.why}
          </p>

          <h4>Properties (from the department dataset)</h4>
          <table>
            <tbody>
              {PROPS.map(([k, label, unit]) => (
                <tr key={k}>
                  <td>{label}</td>
                  <td>
                    {r.properties[k] ?? "—"} {unit}
                    {k === "A5" && r.a5_imputed && <em className="muted"> (filled with family median)</em>}
                  </td>
                </tr>
              ))}
              <tr>
                <td>Corrosion score</td>
                <td>{r.corrosion_score}/5</td>
              </tr>
              <tr>
                <td>Dataset label</td>
                <td>{r.labelled_classes}</td>
              </tr>
            </tbody>
          </table>
        </WhyModal>
      )}
    </article>
  );
}

// "Why this material?" details in a popup (Esc or click outside to close)
function WhyModal({ r, onClose, children }) {
  useEffect(() => {
    const k = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return createPortal(
    <div className="modal-back" onClick={onClose}>
      <div className="modal why-modal" onClick={(e) => e.stopPropagation()}>
        <div className="why-head">
          <div>
            <span className="muted small mono">#{String(r.rank).padStart(2, "0")} · WHY THIS MATERIAL</span>
            <h3>
            {r.display_name}
            {r.rank === 1 && <span className="badge best inline">Best overall</span>}
          </h3>
            <p className="muted small">{r.material_group_label} · {r.standard}</p>
          </div>
          <button className="qclose" onClick={onClose} aria-label="Close">✕</button>
        </div>
        <div className="why">{children}</div>
      </div>
    </div>,
    document.body
  );
}
