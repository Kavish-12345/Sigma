import { useMemo, useState } from "react";

// Ashby-style property chart (log-log): tensile strength vs density for every material
// in the dataset, coloured by family. Optional `highlight` = material names to mark.
export const FAMILY_COLORS = {
  ferrous: "#60a5fa",
  aluminium: "#22d3ee",
  copper: "#f59e0b",
  magnesium: "#c084fc",
};
const FAMILY_NAMES = { ferrous: "Steels & cast irons", aluminium: "Aluminium alloys", copper: "Copper alloys", magnesium: "Magnesium alloys" };

const X_MIN = 1500, X_MAX = 10000, Y_MIN = 50, Y_MAX = 3000;
const W = 760, H = 440, PAD = { l: 64, r: 20, t: 18, b: 48 };
const lx = (v) => PAD.l + ((Math.log10(v) - Math.log10(X_MIN)) / (Math.log10(X_MAX) - Math.log10(X_MIN))) * (W - PAD.l - PAD.r);
const ly = (v) => H - PAD.b - ((Math.log10(v) - Math.log10(Y_MIN)) / (Math.log10(Y_MAX) - Math.log10(Y_MIN))) * (H - PAD.t - PAD.b);

const X_TICKS = [1500, 2000, 3000, 5000, 7000, 10000];
const Y_TICKS = [50, 100, 200, 500, 1000, 2000, 3000];

// yKey: "Su" or "Sy". indexLine: {n, text, through: result} draws M = y^n/ρ = const through that material.
// limit: design requirements; the band below limit[yKey] is shaded (screened out).
export default function AshbyChart({ materials = [], highlight = [], compact = false, yKey = "Su", indexLine = null, limit = null }) {
  const [hover, setHover] = useState(null);
  const [hidden, setHidden] = useState([]);
  const hl = useMemo(() => new Set(highlight), [highlight]);
  const yName = yKey === "Sy" ? "YIELD STRENGTH Sy" : "TENSILE STRENGTH Su";

  const pts = useMemo(
    () =>
      materials
        .filter((m) => m[yKey] > 0 && m.Ro > 0 && m.Ro >= X_MIN && m.Ro <= X_MAX)
        .map((m) => ({ ...m, Y: Math.min(Math.max(m[yKey], Y_MIN), Y_MAX) })),
    [materials, yKey]
  );

  // selection line: y = (M·ρ)^(1/n), through the #1 material
  let line = null;
  const p0 = indexLine?.through?.properties;
  if (p0 && p0[yKey] > 0 && p0.Ro > 0) {
    const M = Math.pow(p0[yKey], indexLine.n) / p0.Ro;
    const yAt = (ro) => Math.pow(M * ro, 1 / indexLine.n);
    const seg = [];
    for (let i = 0; i <= 40; i++) {
      const ro = X_MIN * Math.pow(X_MAX / X_MIN, i / 40);
      const y = yAt(ro);
      if (y >= Y_MIN && y <= Y_MAX) seg.push([lx(ro), ly(y)]);
    }
    if (seg.length > 1) line = { d: "M" + seg.map((p) => p.join(",")).join(" L"), end: seg[seg.length - 1] };
  }
  const lim = limit?.[yKey];
  const shown = pts.filter((m) => !hidden.includes(m.family) && !hl.has(m.Material));
  const marked = pts.filter((m) => hl.has(m.Material));
  // one label per highlighted material name
  const labelled = [];
  const seen = new Set();
  marked.forEach((m) => {
    if (!seen.has(m.Material)) {
      seen.add(m.Material);
      labelled.push(m);
    }
  });

  const toggle = (f) => setHidden((h) => (h.includes(f) ? h.filter((x) => x !== f) : [...h, f]));

  return (
    <div className={`ashby ${compact ? "compact" : ""}`}>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Tensile strength versus density for all materials">
        {/* grid */}
        {X_TICKS.map((t) => (
          <g key={`x${t}`}>
            <line x1={lx(t)} x2={lx(t)} y1={PAD.t} y2={H - PAD.b} className="ash-grid" />
            <text x={lx(t)} y={H - PAD.b + 18} textAnchor="middle" className="ash-tick">{t.toLocaleString()}</text>
          </g>
        ))}
        {Y_TICKS.map((t) => (
          <g key={`y${t}`}>
            <line x1={PAD.l} x2={W - PAD.r} y1={ly(t)} y2={ly(t)} className="ash-grid" />
            <text x={PAD.l - 8} y={ly(t) + 4} textAnchor="end" className="ash-tick">{t}</text>
          </g>
        ))}
        <text x={(W + PAD.l) / 2} y={H - 8} textAnchor="middle" className="ash-axis">DENSITY ρ (kg/m³) · log scale</text>
        <text x={16} y={(H - PAD.b + PAD.t) / 2} textAnchor="middle" className="ash-axis" transform={`rotate(-90 16 ${(H - PAD.b + PAD.t) / 2})`}>
          {yName} (MPa) · log
        </text>
        {lim > Y_MIN && (
          <g>
            <rect x={PAD.l} y={ly(Math.min(lim, Y_MAX))} width={W - PAD.l - PAD.r} height={ly(Y_MIN) - ly(Math.min(lim, Y_MAX))} className="ash-screen" />
            <text x={W - PAD.r - 6} y={ly(Math.min(lim, Y_MAX)) + 14} textAnchor="end" className="ash-guide-text">
              screened out: {yKey} &lt; {Math.round(lim)} MPa
            </text>
          </g>
        )}
        {line ? (
          <g>
            <path d={line.d} className="ash-index" />
            <text x={line.end[0] - 6} y={line.end[1] + 16} textAnchor="end" className="ash-index-text">
              M = {indexLine.text} = const · better ↖
            </text>
          </g>
        ) : (
          <g>
            {/* specific-strength guide line (Su/ρ constant) */}
            <line x1={lx(1500)} y1={ly(150)} x2={lx(10000)} y2={ly(1000)} className="ash-guide" />
            <text x={lx(1560)} y={ly(160) - 8} className="ash-guide-text">Su/ρ = const (equal specific strength)</text>
          </g>
        )}

        {shown.map((m, i) => (
          <circle
            key={i}
            cx={lx(m.Ro)}
            cy={ly(m.Y)}
            r={compact ? 2.6 : 3.2}
            fill={FAMILY_COLORS[m.family] || "#888"}
            className={`ash-pt ${marked.length ? "dim" : ""}`}
            onMouseEnter={() => setHover(m)}
            onMouseLeave={() => setHover(null)}
          />
        ))}
        {marked.map((m, i) => (
          <circle key={`h${i}`} cx={lx(m.Ro)} cy={ly(m.Y)} r={6} className="ash-hl" onMouseEnter={() => setHover(m)} onMouseLeave={() => setHover(null)} />
        ))}
        {labelled.slice(0, 5).map((m, i) => (
          <text key={`l${i}`} x={lx(m.Ro) + 9} y={ly(m.Y) - 8 - i * 2} className="ash-hl-label">#{highlight.indexOf(m.Material) + 1} {m.Material.slice(0, 22)}</text>
        ))}

        {hover && (
          <g transform={`translate(${Math.min(lx(hover.Ro) + 10, W - 230)} ${Math.max(ly(hover.Y) - 60, 10)})`} className="ash-tip">
            <rect width="220" height="52" rx="6" />
            <text x="10" y="20" className="t1">{hover.Material.slice(0, 30)}</text>
            <text x="10" y="38" className="t2">{yKey} {hover[yKey]} MPa · ρ {hover.Ro} kg/m³</text>
          </g>
        )}
      </svg>
      <div className="ash-legend">
        {Object.entries(FAMILY_COLORS).map(([f, c]) => (
          <button key={f} className={hidden.includes(f) ? "off" : ""} onClick={() => toggle(f)}>
            <span style={{ background: c }} /> {FAMILY_NAMES[f]}
          </button>
        ))}
        {highlight.length > 0 && (
          <span className="ash-legend-hl">
            <span /> Your shortlist
          </span>
        )}
      </div>
    </div>
  );
}
