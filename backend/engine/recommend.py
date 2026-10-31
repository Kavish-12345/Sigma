"""
Recommendation engine.

form -> target classes + weights -> filter -> ML suitability -> price -> TOPSIS
     -> ranked results with a "Why this material?" breakdown.

The ML model makes the selection signal (suitability). No LLM is involved here;
the chat and the No-AI form both call this same function.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .design import FUNCTIONS, calculate, material_index, translation
from .options import FAMILIES, FIELDS, MIN_LIMIT_FIELDS, PRIORITY_WEIGHT, target_classes, topsis_weights
from .preprocess import build_X, clean
from .pricing import cost_value, effective_prices, load_prices, price_category
from .topsis import topsis

BACKEND = Path(__file__).resolve().parents[1]
DATA_DIR = BACKEND / "data"
MODEL_DIR = BACKEND / "models"


# --------------------------------------------------------------------------
# Loading (once per process)
# --------------------------------------------------------------------------
def _proba(model, X) -> np.ndarray:
    P = model.predict_proba(X)
    if isinstance(P, list):
        clf = model[-1] if hasattr(model, "steps") else model
        cols = []
        for p, cls in zip(P, clf.classes_):
            cls = list(cls)
            cols.append(p[:, cls.index(1)] if 1 in cls else np.zeros(len(X)))
        return np.column_stack(cols)
    return np.asarray(P)


@lru_cache(maxsize=1)
def load_engine():
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    model = joblib.load(MODEL_DIR / "suitability_model.joblib")
    mlb = joblib.load(MODEL_DIR / "label_binarizer.joblib")
    prices = load_prices()

    raw = pd.read_csv(DATA_DIR / "Data_labelled.csv")
    df, _ = clean(raw, a5_medians=meta["a5_medians"])
    df = df[df["status"] != "exclude"].reset_index(drop=True)  # out of scope (conductors etc.)

    P = _proba(model, build_X(df, meta["features"]))
    classes = list(mlb.classes_)
    for i, c in enumerate(classes):
        df[f"p::{c}"] = P[:, i]

    df["price_category"] = df.apply(price_category, axis=1)
    cats = prices["categories"]
    df["corrosion_score"] = df["price_category"].map(lambda k: cats[k]["corrosion_score"])
    df["display_name"] = [
        f"{m}" + (f" ({h})" if isinstance(h, str) and h.strip() else "")
        for m, h in zip(df["Material"], df["Heat treatment"])
    ]
    return {"meta": meta, "df": df, "classes": classes, "prices": prices}


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_form(form: dict, prices: dict, classes: list[str]) -> dict:
    """Return a cleaned copy; raise ValueError with a clear message on bad input."""
    f = dict(form)
    props = f.get("properties") or ["not_sure"]
    if isinstance(props, str):
        props = [props]
    for v in props:
        if v != "not_sure" and v not in classes:
            raise ValueError(f"properties: unknown class '{v}'. Model classes: {classes}")
    props = [v for v in props if v != "not_sure"] or ["not_sure"]
    f["properties"] = props
    primary = f.get("primary_property")
    if primary in (None, "", "not_sure") or primary not in props:
        primary = props[0] if len(props) == 1 and props[0] != "not_sure" else None
    f["primary_property"] = primary
    for field in ("environment", "weight_priority", "cost_priority"):
        v = f.get(field) or "not_sure"
        if v not in FIELDS[field]:
            raise ValueError(f"{field}: unknown option '{v}'")
        f[field] = v
    vol = f.get("volume_cm3")
    f["volume_cm3"] = float(vol) if vol not in (None, "", 0) else None
    if f["volume_cm3"] is not None and f["volume_cm3"] <= 0:
        raise ValueError("volume_cm3 must be positive")
    limits = {k: float(v) for k, v in (f.get("min_limits") or {}).items() if v not in (None, "")}
    for k in limits:
        if k not in MIN_LIMIT_FIELDS:
            raise ValueError(f"min_limits: unknown field '{k}'")
    f["min_limits"] = limits
    groups = set(FAMILIES) | set(prices["categories"])
    for key in ("include", "avoid"):
        vals = f.get(key) or []
        bad = [v for v in vals if v not in groups]
        if bad:
            raise ValueError(f"{key}: unknown material group(s) {bad}")
        f[key] = list(vals)
    f["price_overrides"] = {k: float(v) for k, v in (f.get("price_overrides") or {}).items()
                            if k in prices["categories"] and v not in (None, "")}
    f["top_n"] = int(f.get("top_n") or 10)
    fn = f.get("function") or "not_sure"
    if fn != "not_sure" and fn not in FUNCTIONS:
        raise ValueError(f"function: unknown option '{fn}'. Allowed: {list(FUNCTIONS)} or not_sure")
    f["function"] = fn
    di = f.get("design_inputs")
    f["design_inputs"] = di if (fn != "not_sure" and isinstance(di, dict) and di) else None
    return f


ENV_CLASS = {"water_marine": "Corrosion-resistant", "chemical": "Corrosion-resistant",
             "high_temperature": "Heat-resistant"}
ENV_MIN_CORROSION = {"water_marine": 4, "chemical": 4}  # corrosion score (1-5) needed to pass screening


def design_objective(form: dict) -> str:
    """Ashby objective from the priority answers: minimise mass if weight matters more than cost."""
    w = PRIORITY_WEIGHT[form.get("weight_priority", "not_sure")]
    c = PRIORITY_WEIGHT[form.get("cost_priority", "not_sure")]
    return "min_weight" if w > c else "min_cost"


def _design_block(form: dict) -> dict | None:
    """Translation + design calculation, or None when the user skipped the design step."""
    fn = form["function"]
    if fn == "not_sure" or not form["design_inputs"]:
        return None
    calc = calculate(fn, form["design_inputs"])   # raises ValueError on bad inputs
    objective = design_objective(form)
    env = FIELDS["environment"][form["environment"]]["label"]
    return {
        "function": fn,
        "function_label": FUNCTIONS[fn]["label"],
        "translation": translation(fn, calc["inputs"], "Minimise mass" if objective == "min_weight"
                                   else "Minimise material cost", env),
        **calc,
        "index": material_index(fn, objective),
    }


# --------------------------------------------------------------------------
# Main entry
# --------------------------------------------------------------------------
def recommend(form: dict) -> dict:
    eng = load_engine()
    df, prices = eng["df"], eng["prices"]
    form = validate_form(form, prices, eng["classes"])

    design = _design_block(form)
    targets = target_classes(form)
    targets_from_function = False
    if not targets and design:  # properties "not sure": use the classes the part's function implies
        targets = {c: 2 for c in FUNCTIONS[design["function"]]["classes"] if c in eng["classes"]}
        env_class = ENV_CLASS.get(form["environment"])  # the environment is a constraint too
        if env_class in eng["classes"]:
            targets[env_class] = 3
        targets_from_function = True
    raw_w = topsis_weights(form)
    if design:  # Ashby material index: as important as ML suitability
        raw_w["index"] = raw_w["suitability"]
    crit_keys = list(raw_w)

    # 1. Screening: hard limits (user + design calculation) and include / avoid
    w_raw = np.array([raw_w[k] for k in crit_keys], dtype=float)
    empty = {"form": form, "targets": targets, "weights": raw_w, "results": [], "n_candidates": 0,
             "n_alloys_ranked": 0, "cost_basis": "", "note": "", "design": design,
             "weights_normalised": dict(zip(crit_keys, (w_raw / w_raw.sum()).round(3).tolist()))}
    clash = sorted(set(form["include"]) & set(form["avoid"]))
    if clash:
        return {**empty, "problem_fields": ["include", "avoid"],
                "message": f"{', '.join(clash)} is in both 'include' and 'avoid', so no material is left. "
                           "Remove it from one of the two lists."}
    cand = df
    lim = dict(form["min_limits"])
    from_design = set()
    for k, v in (design["requirements"] if design else {}).items():
        if v > lim.get(k, 0):
            lim[k] = v
            from_design.add(k)
    names = {"Su": "tensile strength", "Sy": "yield strength", "A5": "elongation", "E": "Young's modulus"}
    units = {"Su": "MPa", "Sy": "MPa", "A5": "%", "E": "MPa"}
    for col in ("Su", "Sy", "A5", "E"):
        if col in lim:
            before = cand
            cand = cand[cand[col] >= lim[col]]
            if cand.empty:
                if col in from_design:
                    return {**empty, "problem_fields": ["design_inputs"],
                            "message": f"The design calculation needs {names[col]} >= {lim[col]:g} {units[col]}, but the "
                                       f"strongest remaining material has {before[col].max():g} {units[col]}. "
                                       "Increase the section size, reduce the load or lower the factor of safety."}
                return {**empty, "problem_fields": ["min_limits"], "message": f"No material has {names[col]} >= {lim[col]:g} {units[col]} "
                                            f"(highest among the remaining candidates: {before[col].max():g} {units[col]}). "
                                            "Lower this limit."}
    if "Ro_max" in lim:
        before = cand
        cand = cand[cand["Ro"] <= lim["Ro_max"]]
        if cand.empty:
            return {**empty, "problem_fields": ["min_limits"], "message": f"No material has density <= {lim['Ro_max']:g} kg/m3 (lightest remaining: "
                                        f"{before['Ro'].min():g} kg/m3). Density is in kg/m3, e.g. 3 g/cm3 = 3000 kg/m3."}
    min_corr = ENV_MIN_CORROSION.get(form["environment"]) if design else None
    if min_corr:  # Ashby screening: the environment is a hard constraint in the design process
        cand = cand[cand["corrosion_score"] >= min_corr]
        if cand.empty:
            return {**empty, "problem_fields": ["environment"],
                    "message": "No material that meets the design calculation also resists this environment. "
                               "Increase the section size or choose a different environment."}
    in_group = lambda vals: cand["family"].isin(vals) | cand["price_category"].isin(vals)
    if form["include"]:
        cand = cand[in_group(form["include"])]
        if cand.empty:
            return {**empty, "problem_fields": ["include", "min_limits"], "message": "No material in the 'include' list meets the other limits. "
                                        "Add more groups or relax a limit."}
    if form["avoid"]:
        cand = cand[~in_group(form["avoid"])]
        if cand.empty:
            return {**empty, "problem_fields": ["avoid"], "message": "The 'avoid' list removes every remaining material. Avoid fewer groups."}
    n_after_filter = len(cand)
    cand = cand.copy()

    # 2. ML suitability = importance-weighted mean probability of the chosen classes.
    #    If the user is not sure, use the best fit over all classes the model predicts.
    if targets:
        tw = np.array(list(targets.values()), dtype=float)
        P = cand[[f"p::{c}" for c in targets]].to_numpy()
        cand["suitability"] = (P * tw).sum(axis=1) / tw.sum()
    else:
        cand["suitability"] = cand[[f"p::{c}" for c in eng["classes"]]].max(axis=1)

    # 3. Price layer: cited range per group; ranking uses the midpoint
    eff = effective_prices(prices, form["price_overrides"])
    vol = form["volume_cm3"]
    g = cand["price_category"]
    for k in ("low", "typical", "high"):
        cand[f"cost_{k}"] = [cost_value(eff[c][k], r, vol) for c, r in zip(g, cand["Ro"])]
    cand["price_available"] = g.map(lambda c: eff[c]["available"])

    # 4. One row per alloy: keep its most suitable heat-treatment condition
    cand = (cand.sort_values(["suitability", "cost_typical"], ascending=[False, True], na_position="last")
                .drop_duplicates("group", keep="first"))

    # Materials with no price: ranked as the most expensive candidate (conservative) and flagged
    known = cand["cost_typical"].dropna()
    fill = float(known.max()) if len(known) else 1.0
    cand["cost_rank_value"] = cand["cost_typical"].fillna(fill)

    # Ashby material index M = strength^n / rho  (or / (price * rho) when minimising cost)
    if design:
        ix = design["index"]
        perf = cand[ix["prop"]] ** ix["n"] / (cand["E"] if ix["use_E"] else 1.0)
        if ix["objective"] == "min_cost":
            kg_price = cand["price_category"].map(lambda c: eff[c]["typical"]).astype(float)
            M = perf / (cand["Ro"] * kg_price)
            M = M.fillna(M.min() if M.notna().any() else 0.0)  # no price: ranked as the worst (conservative)
        else:
            M = perf / cand["Ro"]
        cand["index_value"] = M
        best = float(M.max()) or 1.0
        cand["index_rel"] = M / best * 100

    # 5. TOPSIS over distinct alloys
    col = {"suitability": "suitability", "cost": "cost_rank_value", "density": "Ro", "corrosion": "corrosion_score",
           "index": "index_value"}
    crit = [col[k] for k in crit_keys]
    w = np.array([raw_w[k] for k in crit_keys], dtype=float)
    benefit = np.array([k in ("suitability", "corrosion", "index") for k in crit_keys])
    cand["topsis"] = topsis(cand[crit].to_numpy(dtype=float), w, benefit)
    cand = cand.sort_values(["topsis", "suitability"], ascending=False)

    top = cand.head(form["top_n"])
    ref = top.iloc[0]["cost_typical"]  # #1's typical cost, for the comparison label
    results = [_explain(row, rank, targets, eff, vol, ref, design)
               for rank, (_, row) in enumerate(top.iterrows(), start=1)]
    if design:
        design = {**design, "screening": {k: lim[k] for k in sorted(from_design)},
                  "min_corrosion_score": min_corr,
                  "targets_from_function": targets_from_function,
                  "n_screened_out": int(len(df) - n_after_filter)}
    return {
        "form": form,
        "design": design,
        "targets": targets,
        "weights": raw_w,
        "weights_normalised": dict(zip(crit_keys, (w / w.sum()).round(3).tolist())),
        "cost_basis": "Rs per part" if vol else "Rs per cm3 of material (enter a part volume for Rs per part)",
        "n_candidates": int(n_after_filter),
        "n_alloys_ranked": int(len(cand)),
        "results": results,
        "note": "TOPSIS scores are relative to the candidate set; changing the set can change the order (rank reversal).",
    }


def _compare(cost, ref, rank):
    if rank == 1:
        return "Top-ranked material (reference)"
    if cost is None or ref is None or pd.isna(cost) or pd.isna(ref) or ref == 0:
        return "Cannot compare (price not available)"
    r = cost / ref
    if 0.9 <= r <= 1.1:
        return "Similar cost to #1"
    return f"About {r:.1f}x the cost of #1" if r > 1 else f"About {r:.1f}x the cost of #1 (cheaper)"


def _design_check(row, design) -> list[dict]:
    """Actual factor of safety of this material for each design requirement."""
    out = []
    for c in design["checks"]:
        have = float(row[c["prop"]])
        if c.get("no_fos"):
            margin = have / c["required"] if c["required"] else None
            out.append({**c, "actual": round(have), "fos": None, "margin": round(margin, 2) if margin else None,
                        "ok": have >= c["required"]})
        else:
            fos = design["fos"] * have / c["required"] if c["required"] else None
            out.append({**c, "actual": round(have), "fos": round(fos, 2) if fos else None,
                        "ok": have >= c["required"]})
    return out


def _explain(row, rank, targets, eff, vol, ref, design=None) -> dict:
    cat = row["price_category"]
    price = eff[cat]
    class_probs = {c: round(float(row[f"p::{c}"]) * 100, 1) for c in targets}
    unit = "per part" if vol else "per cm3"
    if price["available"]:
        lo, hi = float(row["cost_low"]), float(row["cost_high"])
        dec = 2 if hi < 10 else 0
        cost_text = (f"Rs {lo:.{dec}f} {unit} (your price)" if price["custom"]
                     else f"Rs {lo:.{dec}f} - {hi:.{dec}f} {unit}")
        price_text = (f"Rs {price['low']:.0f}/kg (your price)" if price["custom"]
                      else f"Rs {price['low']:.0f} - {price['high']:.0f}/kg")
    else:
        cost_text = price_text = "Price not available - enter your price"
    why = [f"{c}: model suitability {p:.0f}%" for c, p in sorted(class_probs.items(), key=lambda x: -x[1])]
    why.append(f"Density {row['Ro']:.0f} kg/m³, corrosion score {int(row['corrosion_score'])}/5")
    extra = {}
    if design:
        checks = _design_check(row, design)
        extra = {"design_check": checks,
                 "min_fos": min((c["fos"] for c in checks if c["fos"] is not None), default=None),
                 "index_value": float(row["index_value"]),
                 "index_rel_pct": round(float(row["index_rel"]), 1)}
        why.append(f"Material index {design['index']['text']}: {row['index_rel']:.0f}% of the best candidate")
    return {
        **extra,
        "rank": rank,
        "material": row["Material"],
        "heat_treatment": None if pd.isna(row["Heat treatment"]) else row["Heat treatment"],
        "display_name": row["display_name"],
        "standard": row["Std"],
        "family": row["family"],
        "material_group": cat,
        "material_group_label": price["label"],
        "suitability_pct": round(float(row["suitability"]) * 100, 1),
        "class_suitability_pct": class_probs,
        "topsis_score": round(float(row["topsis"]), 4),
        "properties": {k: (None if pd.isna(row[k]) else round(float(row[k]), 3)) for k in ("Su", "Sy", "A5", "E", "G", "mu", "Ro")},
        "a5_imputed": bool(row["A5_imputed"]),
        "price": {
            "available": price["available"],
            "custom": price["custom"],
            "rs_per_kg_low": price["low"],
            "rs_per_kg_high": price["high"],
            "text": price_text,
            "basis": price["basis"],
            "source": price["source"],
            "url": price["url"],
            "date": price["date"],
            "why": price["why"],
        },
        "cost": {
            "unit": unit,
            "low": None if pd.isna(row["cost_low"]) else round(float(row["cost_low"]), 2),
            "high": None if pd.isna(row["cost_high"]) else round(float(row["cost_high"]), 2),
            "text": cost_text,
            "vs_top": _compare(row["cost_typical"], ref, rank),
        },
        "corrosion_score": int(row["corrosion_score"]),
        "labelled_classes": row["app_classes"] if row["status"] == "train" else "(no label; scored by model)",
        "why": why,
    }
