"""
Part-aware questions: the AI explains what matters for THIS part and asks about it.

The wording, the information and the follow-ups come from the LLM, one question at a
time, based on the part and the answers so far. Nothing here is a stored question list.
The LLM still cannot do engineering on its own: every answer must map to an effect the
engine already understands (application classes, environment, priorities, load type),
this module rejects anything else, and values are typed by the user into the design
card. If the LLM fails, the chat falls back to the fixed form.
"""
from __future__ import annotations

import json
import re

from backend.engine.design import COMMON_INPUTS, FUNCTIONS
from backend.engine.fit import USAGE
from backend.engine.options import FIELDS

from .form_state import FIELD_ORDER, _is_none_word, apply_answer

MAX_QUESTIONS = 6
TOPICS = ["the part's role in its machine", "how the load varies (steady, shock, reversing)",
          "the failure it must avoid (wear, fatigue, buckling, corrosion)", "operating temperature and surroundings",
          "contact, sliding and lubrication", "required life and how often it runs",
          "how it will be made and finished", "whether weight or cost matters more for this part"]
# Only one environment fits in the form, so a later, milder answer must not replace a harsher one
ENV_SEVERITY = ["dry_indoor", "outdoor_humid", "high_temperature", "water_marine", "chemical"]
INPUT_KEY = re.compile(r"\b\w+_(mm|kN|Nm|kW|rpm|MPa|N)\b")
UNIT = re.compile(r"\b(kN|N·m|Nm|N|mm|cm|m|kW|W|rpm|MPa|GPa|kg)\b")
CYCLE_FIGURE = re.compile(r"\s*(?:,|;|\bor\b|≈|~)*\s*[≈~]?\s*[\d.,]+\s*(?:[×x]\s*10\S*\s*)?\s*cycles?\b", re.IGNORECASE)


def hint_fits(hint: str, unit: str) -> bool:
    """A hint in another unit would be typed in as it stands: 30 kN into a field in N means 30 N."""
    found = {u.replace("Nm", "N·m") for u in UNIT.findall(hint)}
    return not unit or not found or unit in found


def needs_details(form: dict) -> bool:
    return bool(form.get("part_name")) and not form.get("details_done") and form.get("design_inputs") is None


def clean_effect(raw, classes: list[str]) -> dict:
    """Keep only what the engine understands. Everything else the LLM wrote is dropped."""
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    picked = raw.get("classes")
    good = [c for c in picked if c in classes] if isinstance(picked, list) else []
    if good:
        out["classes"] = good
    for f in ("environment", "weight_priority", "cost_priority"):
        if isinstance(raw.get(f), str) and raw[f] in FIELDS[f] and raw[f] != "not_sure":
            out[f] = raw[f]
    if isinstance(raw.get("loading"), str) and raw["loading"] in ("steady", "cyclic", "shock"):
        out["loading"] = raw["loading"]
    note = str(raw.get("note") or "").strip()[:200]
    if note:
        out["note"] = note
    return out


def _no_cycle_figure(text) -> str:
    """The LLM's cycle counts are not reliable and feed no calculation, so they are never shown."""
    return re.sub(r"\(\s*\)", "", CYCLE_FIGURE.sub("", str(text))).strip()


def clean_question(raw, classes: list[str]) -> dict | None:
    """A usable question has at least two options, and at least one must change the result."""
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("question") or "").strip()[:160]
    opts = []
    for o in (raw.get("options") if isinstance(raw.get("options"), list) else [])[:6]:
        if isinstance(o, dict) and str(o.get("label") or "").strip():
            opts.append({"value": str(len(opts)), "label": _no_cycle_figure(o["label"])[:70],
                         "description": _no_cycle_figure(o.get("description") or "")[:120],
                         "effect": clean_effect(o.get("effect"), classes)})
    for o in opts:  # a priority can only come from a question that asked about it
        for word, field in (("weight", "weight_priority"), ("cost", "cost_priority")):
            if word not in text.lower():
                o["effect"].pop(field, None)
    if not text or len(opts) < 2 or not any(o["effect"] for o in opts):
        return None
    return {"question": text, "info": str(raw.get("info") or "").strip()[:400], "options": opts}


def detail_prompt(form: dict, classes: list[str]) -> str:
    asked = len(form["details"])
    fns = "\n".join(f"  {k} = {f['label']} (inputs: {', '.join(i['key'] for i in f['inputs'] + COMMON_INPUTS)})"
                    for k, f in FUNCTIONS.items())
    envs = ", ".join(k for k in FIELDS["environment"] if k != "not_sure")
    prios = ", ".join(k for k in FIELDS["weight_priority"] if k != "not_sure")
    answers = [{"question": d["question"], "answer": d["answer"]} for d in form["details"]]
    must_finish = asked >= MAX_QUESTIONS
    typed = form["details"][-1].get("typed") if form["details"] else False
    return f"""You are an experienced mechanical design engineer helping choose a material for one specific part.
Part: {form['part_name']}
What the user said so far: {json.dumps(form.get('user_words') or {}, ensure_ascii=False)}
Questions already answered: {json.dumps(answers, ensure_ascii=False)}
Topics to cover where they apply to this part: {'; '.join(TOPICS)}

{"You have asked enough. Return the finish JSON now." if must_finish else
 "Ask the single most useful question not answered yet, specific to this part. Use what the answers already "
 "revealed, skip topics that do not apply, and never repeat a question. Return the finish JSON only when the "
 "relevant topics are covered."}

Return one JSON object. To ask a question:
{{"done": false,
  "info": "one or two plain sentences on why this matters for this part, with typical real-world practice",
  "question": "one short question",
  "options": [{{"label": "short answer", "description": "what it means, optional",
               "effect": {{"classes": [], "environment": "", "weight_priority": "", "cost_priority": "", "loading": "", "note": ""}}}}]}}
Use 2 to 5 options. In each effect include only the keys that answer really changes:
- classes: from {json.dumps(classes)}
- environment: one of {envs}
- weight_priority, cost_priority: one of {prios}
- loading: steady, cyclic or shock (repeated impact)
- note: only when the answer raises a specific risk the material data cannot check, name the check to do, for example
  "Confirm yield strength at 150 °C from supplier data". Leave it out for general remarks.
{'Also add "last_answer_effect": the same effect object, for the last typed answer.' if typed else ""}

To finish:
{{"done": true,
  "summary": "one or two sentences on how this part is loaded and what the design must guard against",
  "function": "the closest design element below",
  "part_type": "closest of {', '.join(USAGE)}, or other (rod = anything pushed and pulled along its length)",
  "guide": {{"properties": "which properties matter for this part and why"}}}}
Design elements:
{fns}
Use general when no standard element fits or several loads act together. A part loaded mainly along its length
(connecting rod, pushrod, strut, link, piston rod) is never a beam: use tie for pure push or pull, and general when
bending or other loads act too. Write the guide in plain sentences, never with input keys such as F_kN.

Rules: plain text only. Never name or suggest materials. Never compute results or write formulas. Never ask for
numbers the design inputs collect (power, speed, forces, moments, dimensions, tooth counts): the user enters exact
values in a card afterwards, so your questions are about conditions and choices, not sizes, ratings or load levels
such as low, medium and high. Do not write unit conversions or cycle counts in options: use plain bands such as
"short service life" unless the user gave a figure. Set weight_priority or cost_priority only in a question about
weight or cost. Output JSON only."""


def hints_prompt(form: dict) -> str:
    """One small job for the LLM: typical values for the exact fields the card shows."""
    f = FUNCTIONS[form["function"]]
    fields = "\n".join(f"{i['key']}: {i['label']} ({i.get('unit') or 'no unit'})" for i in f["inputs"] + COMMON_INPUTS
                       if i.get("type") != "select")
    return f"""Give typical real-world values for a {form['part_name']}.
Fields:
{fields}
Return JSON: {{"hints": {{"field_key": "typical value or range, in that field's exact unit"}}}}. Keep each hint under 40
characters. Include only fields where a typical value exists. No formulas, no materials. Output JSON only."""


def set_hints(form: dict, data) -> None:
    f = FUNCTIONS.get(form.get("function"))
    hints = data.get("hints") if isinstance(data, dict) and isinstance(data.get("hints"), dict) else {}
    units = {i["key"]: i.get("unit", "") for i in f["inputs"] + COMMON_INPUTS if i.get("type") != "select"} if f else {}
    form["hints"] = {k: str(v).strip()[:70] for k, v in hints.items()
                     if k in units and v and hint_fits(str(v), units[k])}


def apply_effect(form: dict, effect: dict, groups: list[str], classes: list[str]) -> None:
    form["suggested"] = list(dict.fromkeys(form["suggested"] + effect.get("classes", [])))
    for f in ("weight_priority", "cost_priority"):
        if f in effect:
            apply_answer(form, f, effect[f], "", groups, classes)
    env = effect.get("environment")
    if env and (form["environment"] not in ENV_SEVERITY
                or ENV_SEVERITY.index(env) > ENV_SEVERITY.index(form["environment"])):
        apply_answer(form, "environment", env, "", groups, classes)
    if "loading" in effect and form["design_draft"].get("loading") != "shock":  # as for environment: harshest stays
        form["design_draft"]["loading"] = effect["loading"]
    if effect.get("note") and effect["note"] not in form["notes"]:
        form["notes"].append(effect["note"])


def answer_detail(form: dict, value, label: str, groups: list[str], classes: list[str]) -> dict:
    """Store the answer to the open question. A typed answer matching no option gets its effect from the next LLM call."""
    q = form["pending_detail"]
    form["pending_detail"] = None
    text = str(label or value or "").strip()
    opt = next((o for o in q["options"] if o["value"] == str(value) or o["label"].lower() == text.lower()), None)
    entry = {"question": q["question"], "answer": opt["label"] if opt else text[:200],
             "typed": not opt and not _is_none_word(text)}
    form["details"].append(entry)
    if opt:
        apply_effect(form, opt["effect"], groups, classes)
    return entry


def finish(form: dict, data: dict, groups: list[str], classes: list[str]) -> str:
    """End of the questions: record the design element and keep the part-specific guidance for the cards."""
    form["details_done"], form["pending_detail"] = True, None
    kind = data.get("part_type")
    form["part_type"] = kind if isinstance(kind, str) and kind in USAGE else None
    fn = data.get("function")
    if fn in FUNCTIONS and form.get("function") is None:
        draft = dict(form["design_draft"])   # recording the function clears the draft
        apply_answer(form, "function", fn, "", groups, classes)
        form["design_draft"] = draft
    guide = data.get("guide") if isinstance(data.get("guide"), dict) else {}
    form["guide"] = {k: str(v).strip()[:300] for k, v in guide.items()
                     if k in FIELD_ORDER and k != "design_inputs" and v and not INPUT_KEY.search(str(v))}
    f = FUNCTIONS.get(form.get("function"))
    if f:  # written here from the real fields, so it can never ask for something the card does not have
        keys = {i["key"] for i in f["inputs"]}
        form["guide"]["design_inputs"] = "Typical values for this part are shown in each field." + (
            " If the part can be pushed along its length, enter the unsupported length and end condition so buckling is checked."
            if "L_mm" in keys and "ends" in keys else "")
    return str(data.get("summary") or "").strip()[:400]


def apply_step(form: dict, data, groups: list[str], classes: list[str]) -> str | None:
    """Use one LLM reply: open the next question, or finish. None = the reply was unusable."""
    if not isinstance(data, dict):
        return None
    if form["details"] and form["details"][-1].get("typed"):
        form["details"][-1]["typed"] = False
        apply_effect(form, clean_effect(data.get("last_answer_effect"), classes), groups, classes)
    if data.get("done") or len(form["details"]) >= MAX_QUESTIONS:
        return finish(form, data, groups, classes) if data.get("done") else None
    q = clean_question(data, classes)
    if q is None:
        return None
    form["pending_detail"] = q
    return q["info"]


def detail_spec(form: dict) -> dict:
    """The open question as a normal single-choice card."""
    q = form["pending_detail"]
    return {"field": "part_detail", "title": q["question"], "type": "single", "optional": False,
            "subtitle": f"About your {form['part_name']}"[:80], "step": len(form["details"]) + 1, "total": MAX_QUESTIONS,
            "options": [{k: o[k] for k in ("value", "label", "description")} for o in q["options"]]}
