import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { getMaterialsSummary } from "../api.js";
import AshbyChart from "../components/AshbyChart.jsx";
import { Bearing, GearTrain, Logo, Piston, WanderingRocket } from "../components/mech.jsx";

const EXAMPLES = [
  { code: "PRT-01", title: "Gear shaft", text: "A hardened gear shaft for an industrial gearbox, machined from bar" },
  { code: "PRT-02", title: "Marine pump housing", text: "A cast pump housing that sits in sea water" },
  { code: "PRT-03", title: "Compression spring", text: "A compression spring for a valve, dry indoors" },
  { code: "PRT-04", title: "Drone frame bracket", text: "A lightweight load-bearing bracket for a drone frame" },
  { code: "PRT-05", title: "Sliding bushing", text: "A bushing that slides on a steel shaft, used outdoors" },
  { code: "PRT-06", title: "Bulk-machined screw", text: "Screws machined in large numbers, cost is critical" },
];

const STEPS = [
  { n: "01", t: "Requirements", d: "Chat or form. You describe the part; nothing is assumed." },
  { n: "02", t: "ML suitability", d: "Gradient Boosting scores every material for the classes you chose." },
  { n: "03", t: "Filters & cost", d: "Hard limits, include/avoid, cited Indian price ranges, cost per part." },
  { n: "04", t: "TOPSIS ranking", d: "Suitability, cost, weight and corrosion balanced by your priorities." },
  { n: "05", t: "Shortlist", d: "Ranked materials with datasheets and a 'why this material' breakdown." },
];

// page: "home" | "how" | "chart" | "model"
export default function Home({ page = "home", onStart, go }) {
  const [summary, setSummary] = useState(null);
  const eyebrowRef = useRef(null);
  const selectRef = useRef(null);

  // stretch "MATERIAL SELECTION" so it spans exactly the width of the word "SELECT" below it
  useLayoutEffect(() => {
    const fit = () => {
      const eb = eyebrowRef.current, word = selectRef.current;
      if (!eb || !word) return;
      eb.style.letterSpacing = "0px";
      eb.style.fontSize = "";
      const n = eb.textContent.length;
      let natural = eb.getBoundingClientRect().width;
      // big condensed glyphs have side bearings: nudge right so the M sits under the S stem
      const nudge = Math.round(word.getBoundingClientRect().height * 0.07);
      eb.style.marginLeft = `${nudge}px`;
      const target = word.getBoundingClientRect().width - nudge * 2;
      if (natural > target) {
        eb.style.fontSize = `${parseFloat(getComputedStyle(eb).fontSize) * (target / natural)}px`;
        natural = eb.getBoundingClientRect().width;
      }
      eb.style.letterSpacing = `${Math.max(0, (target - natural) / (n - 1))}px`;
      eb.style.marginRight = `-${eb.style.letterSpacing}`; // trailing spacing does not count
    };
    fit();
    document.fonts?.ready.then(fit);
    window.addEventListener("resize", fit);
    return () => window.removeEventListener("resize", fit);
  }, [page]);
  useEffect(() => {
    getMaterialsSummary().then(setSummary).catch(() => setSummary(null));
  }, []);
  const s = summary?.model?.scores?.All;
  const link = (to, label) => (
    <a
      href={to}
      className={(to === "/how" && page === "how") || (to === "/chart" && page === "chart") || (to === "/model" && page === "model") ? "active" : ""}
      onClick={(e) => {
        e.preventDefault();
        go(to);
      }}
    >
      {label}
    </a>
  );

  return (
    <div className={`home ${page === "home" ? "landing" : "subpage"}`}>
      <header className="home-nav">
        <div className="brand" onClick={() => go("/")} title="Home">
          <Logo size={34} />
          <div>
            <h1 className="wordmark">SIGMA</h1>
            <p>Material selection engine</p>
          </div>
        </div>
        <nav>
          {link("/how", "How it works")}
          {link("/chart", "Property chart")}
          {link("/model", "Model")}
        </nav>
      </header>

      {page === "home" && (
      <div className="landing-body">
      <div className="rocket-layer top"><WanderingRocket /></div>
      <section className="hero">
        <div className="hero-text">
          <div className="eyebrow"><span ref={eyebrowRef} className="eyebrow-fit">MATERIAL SELECTION</span></div>
          <h2>
            <span ref={selectRef}>Select</span> the right material.
            <br />
            <span className="hl">Engineered by data.</span>
          </h2>
          <p className="hero-line">
            <span className="brace">{"{"}</span> Describe your part. Get a ranked, costed material shortlist. <span className="brace">{"}"}</span>
          </p>
          <div className="hero-cta">
            <button className="primary big" onClick={() => onStart({ mode: "chat" })}>Start with the assistant</button>
            <button className="ghost big" onClick={() => onStart({ mode: "form" })}>Use the spec form</button>
          </div>
        </div>
        <div className="hero-art">
          <GearTrain />
          <div className="hero-side">
            <Piston className="hero-piston" />
            <Bearing className="hero-bearing" />
          </div>
        </div>
      </section>

      </div>
      )}

      {page === "how" && (
      <>
      <section id="how" className="block">
        <div className="block-head"><span className="sec">SEC. B</span><h3>How it works</h3></div>
        <div className="flow">
          {STEPS.map((st, i) => (
            <div key={st.n} className="flow-step">
              <div className="flow-n">{st.n}</div>
              <div className="flow-t">{st.t}</div>
              <div className="flow-d">{st.d}</div>
              {i < STEPS.length - 1 && <div className="flow-arrow">→</div>}
            </div>
          ))}
        </div>
      </section>

      </>
      )}

      {page === "chart" && (
      <section id="chart" className="block">
        <div className="block-head">
          <span className="sec">SEC. D</span>
          <h3>Strength vs density, every material in the dataset</h3>
          <p className="muted">An Ashby-style property chart. Hover a point; click the legend to hide a family. Points up and to the left are strong for their weight.</p>
        </div>
        <div className="panel bp">{summary ? <AshbyChart materials={summary.materials} /> : <p className="muted">Loading chart…</p>}</div>
      </section>
      )}

      {page === "how" && (
      <section className="block">
        <div className="block-head"><span className="sec">SEC. C</span><h3>Try a part</h3></div>
        <div className="examples-grid">
          {EXAMPLES.map((e) => (
            <button key={e.code} className="part-card" onClick={() => onStart({ mode: "chat", prompt: e.text })}>
              <span className="part-code">{e.code}</span>
              <span className="part-title">{e.title}</span>
              <span className="part-text">{e.text}</span>
              <span className="part-go">Run →</span>
            </button>
          ))}
        </div>
      </section>

      )}

      {page === "model" && (
      <section id="model" className="block">
        <div className="block-head"><span className="sec">SEC. E</span><h3>Model & data</h3><p className="muted">What the engine is built on, and how well it performs on alloys it has never seen.</p></div>
        <div className="model-grid">
          <div className="panel bp">
            <h4>Dataset</h4>
            <p className="muted small">Department dataset of standard materials (properties never altered).</p>
            <div className="kv"><span>Standards</span><b>{summary ? summary.standards.join(" · ") : "—"}</b></div>
            <div className="kv"><span>Families</span><b>Steels & cast irons · Aluminium · Copper · Magnesium</b></div>
            <div className="kv"><span>Inputs</span><b>Su, Sy, A5, E, μ, ρ, Sy/Su, heat-treatment condition</b></div>
          </div>
          <div className="panel bp">
            <h4>Suitability model</h4>
            <p className="muted small">{summary?.model?.name || "Gradient Boosting"}, 5-fold grouped cross-validation (tempers of one alloy never split).</p>
            <table className="scores">
              <thead><tr><th></th><th>micro-F1</th><th>macro-F1</th><th>exact</th></tr></thead>
              <tbody>
                {summary &&
                  Object.entries(summary.model.scores).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td>{v.micro_F1}</td><td>{v.macro_F1}</td><td>{v.exact_match}</td></tr>
                  ))}
              </tbody>
            </table>
          </div>
          <div className="panel bp">
            <h4>Application classes</h4>
            <div className="chips">{summary?.classes.map((c) => <span key={c} className="chip static">{c}</span>)}</div>
            <p className="muted small" style={{ marginTop: 10 }}>
              Limits: no composition or temperature data, so free-machining and heat resistance are judged from labels and mechanical properties only.
            </p>
          </div>
        </div>
      </section>

      )}

      {page !== "home" && (
      <footer className="titleblock">
        <div className="tb-cell wide"><span>PROJECT</span><b>AI-Based Material Selection for Mechanical Component Design using ML</b></div>
        <div className="tb-cell"><span>DEPARTMENT</span><b>Mechanical Engineering, LNMIIT</b></div>
        <div className="tb-cell"><span>GUIDE</span><b>Prof. Dr. Ashok Kr Dargar</b></div>
        <div className="tb-cell wide"><span>TEAM</span><b>Kavish Srivastava · Samarth Jindal · Aashil Bhutra</b></div>
        <div className="tb-cell"><span>SCALE</span><b>NTS</b></div>
        <div className="tb-cell"><span>SHEET</span><b>{{ how: "1", chart: "2", model: "3" }[page]} / 3</b></div>
      </footer>
      )}
    </div>
  );
}
