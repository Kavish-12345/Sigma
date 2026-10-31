"""
The fixed requirement form that the chat fills.

The LLM can only call record_answer(field, value, user_words). This module
validates every value against the allowed options. The code (not the LLM)
decides when the form is complete.

Form value conventions
    None          -> not answered yet
    "not_sure"    -> user said they are not sure (allowed for every field)
    optional fields answered with "none": volume "not_sure", min_limits {}, include/avoid []
"""
from __future__ import annotations

import re

from backend.engine.design import COMMON_INPUTS, FUNCTIONS, MissingInput, clean_inputs
from backend.engine.options import FAMILIES, FIELDS, MIN_LIMIT_FIELDS

# function + design_inputs = Ashby "translation" (what the part does, its loads and sizes)
REQUIRED = ["part_name", "function", "design_inputs", "properties", "primary_property", "environment",
            "weight_priority", "cost_priority"]
OPTIONAL = ["volume_cm3", "min_limits", "include", "avoid"]
FIELD_ORDER = REQUIRED + OPTIONAL

MAX_ANSWER = 2000  # longer text is cut before parsing: the patterns below get very slow on huge input
_NONE_WORDS = {"not_sure", "not sure", "none", "no", "skip", "nothing", "na", "n/a", "no limits", "dont know", "don't know"}


def empty_form() -> dict:
    form = {f: None for f in FIELD_ORDER}
    form["user_words"] = {}
    form["suggested"] = []   # AI suggestions for 'properties' (shown pre-ticked; never counted until confirmed)
    form["design_draft"] = {}  # design inputs given so far (prefills the card until all are known)
    form["details"] = []  # part-specific questions answered so far (see part_detail.py)
    form["pending_detail"] = None  # the part-specific question currently open
    form["details_done"] = False
    form["guide"] = {}  # part-specific explanation for a form field
    form["hints"] = {}  # typical values for the design inputs (guidance only, never filled in)
    form["notes"] = []  # engineering cautions from the answers, shown with the results
    form["part_type"] = None  # kind of part for a General part (see USAGE in fit.py)
    return form


def normalise_form(form: dict | None) -> dict:
    base = empty_form()
    if form:
        for k in FIELD_ORDER:
            if k in form:
                base[k] = form[k]
        base["user_words"] = dict(form.get("user_words") or {})
        base["suggested"] = list(form.get("suggested") or [])
        base["design_draft"] = dict(form.get("design_draft") or {})
        base["details"] = list(form.get("details") or [])
        base["pending_detail"] = form.get("pending_detail")
        base["details_done"] = bool(form.get("details_done"))
        base["guide"] = dict(form.get("guide") or {})
        base["hints"] = dict(form.get("hints") or {})
        base["notes"] = list(form.get("notes") or [])
        base["part_type"] = form.get("part_type")
    return base


def _match_function(text: str) -> str | None:
    t = text.strip().lower().replace(" ", "_")
    if t in FUNCTIONS:
        return t
    for k, f in FUNCTIONS.items():
        if text.strip().lower() == f["label"].lower() or text.strip().lower() == f["label"].split(" (")[0].lower():
            return k
    return None


def _parse_design(raw) -> dict:
    """'P_kW=10, N_rpm=960, loading=cyclic' or a dict -> dict."""
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw)
    if text.strip().startswith("{"):
        import json
        try:
            return dict(json.loads(text))
        except ValueError:
            pass
    return {k: v for k, v in re.findall(r"([A-Za-z_]+)\s*[=:]\s*([A-Za-z_]+|\d+(?:\.\d+)?)", text)}


def missing_fields(form: dict) -> list[str]:
    return [f for f in FIELD_ORDER if form.get(f) is None]


def _is_none_word(v: str) -> bool:
    return v.strip().lower().replace("_", " ") in {w.replace("_", " ") for w in _NONE_WORDS}


def _split(value) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in re.split(r"[,;]", str(value)) if v.strip()]


def _match_classes(vals: list[str], classes: list[str]) -> tuple[list[str], list[str]]:
    lookup = {c.lower(): c for c in classes}
    good, bad = [], []
    for v in vals:
        c = lookup.get(v.strip().lower())
        (good if c else bad).append(c or v)
    return good, bad


def apply_answer(form: dict, field: str, value, user_words: str, groups: list[str],
                 classes: list[str] | None = None) -> str | None:
    """Validate and store one answer. Returns an error message, or None if recorded."""
    classes = classes or []
    if field == "suggested_properties":  # AI hint only: pre-ticks the card, user must confirm
        good, _ = _match_classes(_split(value or ""), classes)
        form["suggested"] = good
        return None
    if field not in FIELD_ORDER:
        return f"Unknown field '{field}'. Valid fields: {', '.join(FIELD_ORDER)}"
    raw = "" if value is None else value
    if isinstance(raw, str):
        raw = raw[:MAX_ANSWER]
    text = raw if isinstance(raw, str) else str(raw)[:MAX_ANSWER]

    if field == "part_name":
        name = text.strip() or str(user_words or "").strip()
        if not name:
            return "part_name must not be empty"
        form[field] = name[:80]

    elif field == "function":
        if _is_none_word(text):
            form["function"], form["design_inputs"] = "not_sure", "not_sure"
        else:
            fn = _match_function(text)
            if not fn:
                return f"Unknown function '{text}'. Allowed: {', '.join(FUNCTIONS)} or not_sure"
            if form.get("function") != fn:
                form["design_inputs"], form["design_draft"] = None, {}
            form["function"] = fn
            if not form.get("suggested"):  # pre-tick the classes this function implies
                form["suggested"] = [c for c in FUNCTIONS[fn]["classes"] if c in classes]

    elif field == "design_inputs":
        fn = form.get("function")
        if fn in (None, "not_sure"):
            if _is_none_word(text) or fn == "not_sure":
                form[field] = "not_sure"
                return None
            return "Record the part's function first"
        if not isinstance(raw, dict) and _is_none_word(text):
            form[field] = "not_sure"
        else:
            keys = {i["key"] for i in FUNCTIONS[fn]["inputs"] + COMMON_INPUTS}
            draft = {**form.get("design_draft", {}),
                     **{k: v for k, v in _parse_design(raw).items() if k in keys and v not in (None, "")}}
            form["design_draft"] = draft
            try:
                form[field] = clean_inputs(fn, draft)
            except MissingInput as e:
                if not isinstance(raw, dict):
                    return f"{e}. Saved so far: {draft}. Ask the user for the missing value(s)."
                form[field] = "not_sure"  # a card submitted as it is: go on without the calculation
                form["notes"].append(f"Stress calculation skipped: {str(e)[0].lower()}{str(e)[1:]}.")
            except ValueError as e:
                return f"{e}. Saved so far: {draft}. Ask the user for the missing value(s)."

    elif field == "properties":
        vals = _split(raw)
        if not vals or (len(vals) == 1 and _is_none_word(vals[0])):
            form["properties"], form["primary_property"] = ["not_sure"], "not_sure"
        else:
            good, bad = _match_classes([v for v in vals if not _is_none_word(v)], classes)
            if bad or not good:
                return f"Unknown property {bad}. Allowed: {', '.join(classes)}"
            form["properties"] = good
            # one class -> it is the most important; several -> ask which one matters most
            if len(good) == 1:
                form["primary_property"] = good[0]
            elif form.get("primary_property") not in good:
                form["primary_property"] = None

    elif field == "primary_property":
        chosen = [p for p in (form.get("properties") or []) if p != "not_sure"]
        if _is_none_word(text):
            form[field] = "not_sure"
        else:
            good, _ = _match_classes([text], chosen)
            if not good:
                return f"primary_property must be one of the chosen properties: {', '.join(chosen) or 'none chosen yet'}"
            form[field] = good[0]

    elif field in ("environment", "weight_priority", "cost_priority"):
        v = text.strip()
        if v not in FIELDS[field]:
            return f"Invalid {field} '{v}'. Allowed: {', '.join(FIELDS[field])}"
        form[field] = v

    elif field == "volume_cm3":
        if _is_none_word(text):
            form[field] = "not_sure"
        else:
            m = re.search(r"\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
            if not m or not 0 < float(m.group()) < 1e6:
                return "volume_cm3 must be a positive, realistic number in m3, or 'not_sure'"
            # Entered in m3 (a unit written in the answer wins); stored in cm3, which the cost formula uses
            unit = text.lower()
            form[field] = float(m.group()) * (1.0 if "cm" in unit else 0.001 if "mm" in unit else 1e6)

    elif field == "min_limits":
        if isinstance(raw, dict):
            pairs = [(k, raw[k]) for k in raw]
        elif _is_none_word(text):
            pairs = []
        else:
            pairs = re.findall(r"([A-Za-z_]+)\s*[=:>]+\s*(\d+(?:\.\d+)?)", text)
            if not pairs:
                return "min_limits format: 'Sy=400, Su=600, A5=10, E=70000, Ro_max=3000' or 'none'"
        out = {}
        for k, v in pairs:
            if k not in MIN_LIMIT_FIELDS:
                return f"Unknown limit '{k}'. Allowed: {', '.join(MIN_LIMIT_FIELDS)}"
            out[k] = float(v)
        err = check_limit_units(out)
        if err:
            return err
        form[field] = out

    elif field in ("include", "avoid"):
        if isinstance(raw, list):
            vals = _split(raw)
        elif _is_none_word(text):
            vals = []
        else:
            vals = _split(raw)
        bad = [v for v in vals if v not in groups]
        if bad:
            return f"Unknown material group(s) {bad}. Allowed: {', '.join(groups)}"
        other = form.get("avoid" if field == "include" else "include") or []
        clash = [v for v in vals if v in other]
        if clash:
            return (f"{clash} is already in '{'avoid' if field == 'include' else 'include'}'. "
                    "A group cannot be both included and avoided; ask the user which one they mean.")
        form[field] = vals

    if user_words:
        form["user_words"][field] = str(user_words)[:200]
    return None


def check_limit_units(lim: dict) -> str | None:
    """Catch common unit mistakes before they silently empty the result list."""
    if "Ro_max" in lim and lim["Ro_max"] < 100:
        return (f"Max density {lim['Ro_max']:g} looks like g/cm3. Density must be in kg/m3 "
                f"(e.g. {lim['Ro_max'] * 1000:g} kg/m3; steel is about 7850, aluminium about 2700).")
    if "Ro_max" in lim and lim["Ro_max"] < 1700:
        return (f"Max density {lim['Ro_max']:g} kg/m3 is lighter than every metal in the dataset "
                "(the lightest, magnesium, is about 1,740 kg/m3; aluminium about 2,700).")
    if "E" in lim and lim["E"] < 1000:
        return (f"E = {lim['E']:g} looks like GPa. E must be in MPa "
                f"(e.g. {lim['E'] * 1000:g} MPa; steel is about 210000).")
    for k in ("Su", "Sy"):
        if k in lim and lim[k] > 3000:
            return f"{k} = {lim[k]:g} MPa is higher than any material in the dataset (about 2000 MPa max)."
    if "A5" in lim and lim["A5"] > 80:
        return f"Elongation A5 = {lim['A5']:g}% is unrealistically high (typical metals 1-60%)."
    return None


def to_recommend_payload(form: dict, price_overrides: dict | None = None, top_n: int = 10) -> dict:
    """Convert the chat form into the /api/recommend request (same one the No-AI form sends)."""
    vol = form.get("volume_cm3")
    return {
        "part_name": form.get("part_name"),
        "function": form.get("function") or "not_sure",
        "design_inputs": form.get("design_inputs") if isinstance(form.get("design_inputs"), dict) else None,
        "properties": form.get("properties") or ["not_sure"],
        "primary_property": form.get("primary_property") if form.get("primary_property") != "not_sure" else None,
        "environment": form.get("environment") or "not_sure",
        "weight_priority": form.get("weight_priority") or "not_sure",
        "cost_priority": form.get("cost_priority") or "not_sure",
        "volume_cm3": vol if isinstance(vol, (int, float)) else None,
        "min_limits": form.get("min_limits") if isinstance(form.get("min_limits"), dict) else {},
        "include": form.get("include") if isinstance(form.get("include"), list) else [],
        "avoid": form.get("avoid") if isinstance(form.get("avoid"), list) else [],
        "part_notes": form.get("notes") or [],
        "part_type": form.get("part_type"),
        "price_overrides": price_overrides or {},
        "top_n": top_n,
    }


def allowed_groups(price_categories: dict) -> list[str]:
    return list(FAMILIES) + list(price_categories)
