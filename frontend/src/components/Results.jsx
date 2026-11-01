import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import ResultCard from "./ResultCard.jsx";
import AshbyChart from "./AshbyChart.jsx";
import DesignPanel from "./DesignPanel.jsx";
import Datasheet, { CompareTable } from "./Datasheet.jsx";
import { groupLabel } from "../labels.js";

export default function Results({ result, summary, options, form, busy, priceOverrides, onOverride, onEdit, onNew }) {
  const overrides = Object.entries(priceOverrides);
  const [sheet, setSheet] = useState(null);
  const [compare, setCompare] = useState([]);
  const [showCompare, setShowCompare] = useState(false);
  const [showChart, setShowChart] = useState(false);
  const [showDesign, setShowDesign] = useState(false);
  const toggleCompare = (name) =>
    setCompare((c) => (c.includes(name) ? c.filter((x) => x !== name) : c.length < 3 ? [...c, name] : c));
  const compareItems = result.results.filter((r) => compare.includes(r.material));
  const notes = result.form?.part_notes || [];

  return (
    <div className="results">
      <div className="results-head">
        <div className="results-name">
          <span className="mono small">RECOMMENDED MATERIALS</span>
          <h2>{form.part_name || "Your part"}</h2>
        </div>
        <div className="row">
          {(result.design || notes.length > 0) && (
            <button className="ghost" onClick={() => setShowDesign(true)}>
              Design details
            </button>
          )}
          {summary && result.results.length > 0 && (
            <button className="ghost" onClick={() => setShowChart(true)}>
              Property chart
            </button>
          )}
          <button className="ghost" onClick={onEdit}>
            Edit requirements
          </button>
          <button className="ghost" onClick={onNew}>
            New search
          </button>
        </div>
      </div>

      {overrides.length > 0 && (
      <div className="explain-strip">
          <div>
            <strong>Your prices</strong>
            <div className="chips">
              {overrides.map(([g, v]) => (
                <span key={g} className="chip custom">
                  {groupLabel(options, g)}: Rs {v}/kg{" "}
                  <button className="x" onClick={() => onOverride(g, null)} title="Remove">
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>
      </div>
      )}

      {result.message && <div className="error">{result.message}</div>}
      {busy && <div className="muted">Re-ranking…</div>}

      <div className="cards">
        {result.results.map((r) => (
          <ResultCard
            key={r.material}
            r={r}
            onOverride={onOverride}
            onDatasheet={() => setSheet(r)}
            compared={compare.includes(r.material)}
            compareFull={compare.length >= 3}
            onCompare={() => toggleCompare(r.material)}
          />
        ))}
      </div>

      {compare.length > 0 && (
        <div className="compare-bar">
          <span className="mono">{compare.length}/3 selected</span>
          <span className="muted small">{compareItems.map((r) => r.material).join(" · ")}</span>
          <button className="primary small" disabled={compare.length < 2} onClick={() => setShowCompare(true)}>
            Compare →
          </button>
          <button className="ghost small" onClick={() => setCompare([])}>Clear</button>
        </div>
      )}

      {sheet && <Datasheet r={sheet} summary={summary} partName={form.part_name} onClose={() => setSheet(null)} />}
      {showDesign && (
        <Popup kicker="DESIGN DETAILS" title={result.design?.function_label || form.part_name} className="design-modal" onClose={() => setShowDesign(false)}>
          <DesignPanel
            design={result.design}
            result={result}
            notes={notes}
            onChart={summary ? () => { setShowDesign(false); setShowChart(true); } : null}
          />
        </Popup>
      )}
      {showChart && (
        <ChartModal onClose={() => setShowChart(false)} design={result.design}>
          <AshbyChart
            materials={summary.materials}
            highlight={result.results.map((r) => r.material)}
            yKey={result.design && result.design.index.prop === "Sy" && !result.design.index.use_E ? "Sy" : "Su"}
            indexLine={
              result.design && !result.design.index.use_E
                ? { n: result.design.index.n, text: result.design.index.mass_text, through: result.results[0] }
                : null
            }
            limit={result.design?.requirements}
          />
        </ChartModal>
      )}
      {showCompare && <CompareTable items={compareItems} summary={summary} onClose={() => setShowCompare(false)} />}
    </div>
  );
}

function Popup({ kicker, title, note, className = "chart-modal", onClose, children }) {
  useEffect(() => {
    const k = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [onClose]);
  return createPortal(
    <div className="modal-back" onClick={onClose}>
      <div className={`modal ${className}`} onClick={(e) => e.stopPropagation()}>
        <div className="why-head">
          <div>
            <span className="muted small mono">{kicker}</span>
            <h3>{title}</h3>
            {note && <p className="muted small">{note}</p>}
          </div>
          <button className="qclose" onClick={onClose} aria-label="Close">✕</button>
        </div>
        {children}
      </div>
    </div>,
    document.body
  );
}

// shortlist on the strength-vs-density chart, in a popup
function ChartModal({ onClose, children, design }) {
  return (
    <Popup
      kicker="PROPERTY CHART"
      title={design ? "Ashby chart with selection line" : "Your shortlist on the property chart"}
      note={
        design
          ? `Dashed line: constant material index M = ${design.index.mass_text} through #1. Materials above the line perform better for this function; the shaded band is removed by the design calculation.`
          : "Recommended materials (orange) among every material in the dataset."
      }
      onClose={onClose}
    >
      {children}
    </Popup>
  );
}
