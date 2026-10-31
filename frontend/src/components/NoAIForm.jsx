import { groupLabel } from "../labels.js";
import { designComplete, designInputs, inputShown } from "../api.js";

// Plain form that fills exactly the same fields as the chat and calls the same /api/recommend.
export default function NoAIForm({ options, form, setForm, onReview }) {
  const set = (k, v) => setForm({ ...form, [k]: v });
  const groups = [...options.families, ...Object.keys(options.material_groups)];

  // properties = the classes the trained model predicts (from /api/options)
  function toggleProperty(v) {
    let cur = form.properties || [];
    if (v === "not_sure") cur = cur.includes("not_sure") ? [] : ["not_sure"];
    else cur = cur.includes(v) ? cur.filter((x) => x !== v) : [...cur.filter((x) => x !== "not_sure"), v];
    const real = cur.filter((x) => x !== "not_sure");
    let primary = form.primary_property;
    if (cur.includes("not_sure")) primary = "not_sure";
    else if (real.length === 1) primary = real[0];
    else if (!real.includes(primary)) primary = null;
    setForm({ ...form, properties: cur.length ? cur : null, primary_property: cur.length ? primary : null });
  }
  const chosen = (form.properties || []).filter((x) => x !== "not_sure");

  // Design step: function -> its input fields. design_inputs is set once every required value is filled.
  const inputs = designInputs(options, form.function);
  const draft = form.design_draft || {};
  function chooseFunction(fn) {
    if (!fn) return setForm({ ...form, function: null, design_inputs: null, design_draft: {} });
    if (fn === "not_sure") return setForm({ ...form, function: "not_sure", design_inputs: "not_sure", design_draft: {} });
    const defaults = {};
    designInputs(options, fn).forEach((i) => "default" in i && (defaults[i.key] = i.default));
    setForm({ ...form, function: fn, design_inputs: null, design_draft: defaults });
  }
  function setDesign(k, v) {
    const next = { ...draft, [k]: v };
    if (v === "") delete next[k];
    const value = {};
    Object.entries(next).forEach(([key, val]) => {
      const spec = inputs.find((i) => i.key === key);
      value[key] = spec?.type === "select" ? val : Number(val);
    });
    setForm({ ...form, design_draft: next, design_inputs: designComplete(inputs, next) ? value : null });
  }

  function setLimit(k, v) {
    const lim = { ...(form.min_limits || {}) };
    if (v === "") delete lim[k];
    else lim[k] = Number(v);
    set("min_limits", lim);
  }

  function toggleGroup(field, g) {
    const cur = form[field] || [];
    set(field, cur.includes(g) ? cur.filter((x) => x !== g) : [...cur, g]);
  }

  // Optional fields left blank count as "none" so the summary can be confirmed.
  function review() {
    const skipped = form.function && !form.design_inputs; // values left blank: go on without the calculation
    setForm({
      ...form,
      design_inputs: skipped ? "not_sure" : form.design_inputs,
      notes: skipped ? ["Stress calculation skipped: not all loads and sizes were entered."] : form.notes,
      volume_cm3: form.volume_cm3 ?? "not_sure",
      min_limits: form.min_limits ?? {},
      include: form.include ?? [],
      avoid: form.avoid ?? [],
    });
    onReview();
  }

  const select = (field, label) => (
    <label className="field">
      <span>{label}</span>
      <select value={form[field] || ""} onChange={(e) => set(field, e.target.value || null)}>
        <option value="">Choose…</option>
        {options.fields[field].map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <div className="noai">
      <h2>Requirements form</h2>
      <p className="muted">
        Same questions as the chat, following material selection in the design process: translation → screening → ranking.
        Choose “Not sure” whenever you don't know.
      </p>

      <label className="field">
        <span>Part name</span>
        <input value={form.part_name || ""} onChange={(e) => set("part_name", e.target.value || null)} placeholder="e.g. Gear shaft" />
      </label>

      <div className="design-step">
        <div className="step-tag mono">STEP 1 · TRANSLATION + DESIGN CALCULATION</div>
        <label className="field">
          <span>What does the part do? (how is it loaded)</span>
          <select value={form.function || ""} onChange={(e) => chooseFunction(e.target.value)}>
            <option value="">Choose…</option>
            {options.fields.function.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
            <option value="not_sure">Not sure (skip the design calculation)</option>
          </select>
          {form.function && form.function !== "not_sure" && (
            <small className="muted">
              {options.fields.function.find((o) => o.value === form.function)?.description} · e.g.{" "}
              {options.fields.function.find((o) => o.value === form.function)?.examples}
            </small>
          )}
        </label>
        {inputs.length > 0 && (
          <div className="grid2">
            {inputs.filter((i) => inputShown(i, draft)).map((i) =>
              i.type === "select" ? (
                <label className="field" key={i.key}>
                  <span>{i.label}</span>
                  <select value={draft[i.key] ?? i.default} onChange={(e) => setDesign(i.key, e.target.value)}>
                    {i.options.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </label>
              ) : (
                <label className="field" key={i.key}>
                  <span>
                    {i.label}
                    {i.unit ? ` (${i.unit})` : ""}
                    {i.optional ? " · optional" : ""}
                  </span>
                  <input type="number" min="0" step="any" value={draft[i.key] ?? ""} placeholder={i.hint}
                    onChange={(e) => setDesign(i.key, e.target.value)} />
                </label>
              )
            )}
          </div>
        )}
        {inputs.length > 0 && !form.design_inputs && <p className="muted small">Fill the values to run the design calculation. If you leave them blank, it is skipped.</p>}
      </div>

      <div className="field">
        <span>Which properties does the part need? (choose one or more)</span>
        {form.function && options.design_functions?.[form.function] && (
          <small className="muted">
            Typical for this function: {options.design_functions[form.function].classes.join(", ")}
          </small>
        )}
        <div className="prop-list">
          {options.fields.properties.map((o) => {
            const on = form.properties?.includes(o.value);
            return (
              <button type="button" key={o.value} className={`prop ${on ? "on" : ""}`} onClick={() => toggleProperty(o.value)}>
                <span className={`qbox ${on ? "on" : ""}`}>{on ? "✓" : ""}</span>
                <span className="qlabel">
                  {o.label}
                  <span className="qdesc">
                    {o.description}
                    {o.examples ? ` · e.g. ${o.examples}` : ""}
                  </span>
                </span>
              </button>
            );
          })}
          <button
            type="button"
            className={`chip ${form.properties?.includes("not_sure") ? "on" : ""}`}
            style={{ alignSelf: "flex-start" }}
            onClick={() => toggleProperty("not_sure")}
          >
            Not sure
          </button>
        </div>
      </div>

      {chosen.length > 1 && (
        <label className="field">
          <span>Which one matters most?</span>
          <select value={form.primary_property || ""} onChange={(e) => set("primary_property", e.target.value || null)}>
            <option value="">Choose…</option>
            {chosen.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
      )}

      <div className="grid2">
        {select("environment", "Environment")}
        {select("weight_priority", "How important is low weight?")}
        {select("cost_priority", "How important is low cost?")}
        <label className="field">
          <span>Part volume (m³, optional)</span>
          <input
            type="number"
            min="0"
            step="any"
            value={typeof form.volume_cm3 === "number" ? Number((form.volume_cm3 / 1e6).toPrecision(6)) : ""}
            onChange={(e) => set("volume_cm3", e.target.value ? Number(e.target.value) * 1e6 : null)}
            placeholder="e.g. 0.00012, for cost per part"
          />
        </label>
      </div>

      <details className="optional">
        <summary>Optional: minimum limits and material groups</summary>
        <div className="grid2">
          {Object.entries(options.min_limits).map(([k, label]) => (
            <label className="field" key={k}>
              <span>{label}</span>
              <input type="number" min="0" value={form.min_limits?.[k] ?? ""} onChange={(e) => setLimit(k, e.target.value)} />
            </label>
          ))}
        </div>
        {["include", "avoid"].map((field) => (
          <div className="field" key={field}>
            <span>{field === "include" ? "Only consider" : "Avoid"}</span>
            <div className="chips">
              {groups.map((g) => (
                <button
                  type="button"
                  key={g}
                  className={`chip small ${form[field]?.includes(g) ? (field === "avoid" ? "off" : "on") : ""}`}
                  onClick={() => toggleGroup(field, g)}
                >
                  {groupLabel(options, g)}
                </button>
              ))}
            </div>
          </div>
        ))}
      </details>

      <button className="primary" onClick={review}>
        Review summary
      </button>
    </div>
  );
}
