"""
Fixed requirement form: allowed options and how answers become ML targets and TOPSIS weights.

The "properties" question is NOT hand-made: its options are the application classes
the trained model predicts (from label_binarizer.joblib), with plain-language text
from backend/data/class_descriptions.json. The user confirms which classes the part
needs and which one matters most, so no answer-to-class mapping table is needed.

Other questions feed TOPSIS weights only (they do not touch the ML model).
"""
from __future__ import annotations

import json
from pathlib import Path

from .design import FUNCTIONS, function_options

DESCRIPTIONS_PATH = Path(__file__).resolve().parents[1] / "data" / "class_descriptions.json"

PRIORITY_WEIGHT = {"critical": 3, "important": 2, "not_important": 0, "not_sure": 1}
PRIMARY_IMPORTANCE, OTHER_IMPORTANCE = 3, 2   # most important class vs the other chosen classes

FAMILIES = ["ferrous", "aluminium", "copper", "magnesium"]

FIELDS = {
    "environment": {   # sets the corrosion criterion weight in TOPSIS
        "dry_indoor":       {"label": "Dry, indoors",                "corrosion_weight": 0},
        "outdoor_humid":    {"label": "Outdoors / humid",            "corrosion_weight": 2},
        "water_marine":     {"label": "Water or sea water",          "corrosion_weight": 3},
        "chemical":         {"label": "Chemicals / acids",           "corrosion_weight": 3},
        "high_temperature": {"label": "High temperature",            "corrosion_weight": 1},
        "not_sure":         {"label": "Not sure",                    "corrosion_weight": 1},
    },
    "weight_priority": {k: {"label": k.replace("_", " ").capitalize()} for k in PRIORITY_WEIGHT},
    "cost_priority":   {k: {"label": k.replace("_", " ").capitalize()} for k in PRIORITY_WEIGHT},
}

MIN_LIMIT_FIELDS = {
    "Su": "Min tensile strength (MPa)",
    "Sy": "Min yield strength (MPa)",
    "A5": "Min elongation (%)",
    "E": "Min Young's modulus (MPa)",
    "Ro_max": "Max density (kg/m³)",
}


def load_descriptions() -> dict:
    try:
        return json.loads(DESCRIPTIONS_PATH.read_text(encoding="utf-8"))["classes"]
    except (OSError, ValueError, KeyError):
        return {}


def property_options(classes: list[str]) -> list[dict]:
    """One option per class the model predicts, in the model's order."""
    desc = load_descriptions()
    return [{"value": c, "label": c, "description": desc.get(c, {}).get("description", ""),
             "examples": desc.get(c, {}).get("examples", "")} for c in classes]


def target_classes(form: dict) -> dict[str, int]:
    """{class: importance}. Empty dict = user not sure -> engine uses best fit over all classes."""
    props = [p for p in (form.get("properties") or []) if p != "not_sure"]
    primary = form.get("primary_property")
    return {p: (PRIMARY_IMPORTANCE if (p == primary or len(props) == 1) else OTHER_IMPORTANCE) for p in props}


def topsis_weights(form: dict) -> dict[str, float]:
    """Raw criterion weights. Suitability is always the highest weight."""
    cost = PRIORITY_WEIGHT[form.get("cost_priority", "not_sure")]
    weight = PRIORITY_WEIGHT[form.get("weight_priority", "not_sure")]
    corr = FIELDS["environment"][form.get("environment", "not_sure")]["corrosion_weight"]
    suit = max(cost, weight, corr, 2) + 1
    return {"suitability": suit, "cost": cost, "density": weight, "corrosion": corr}


def options_payload(price_categories: dict, classes: list[str]) -> dict:
    """What GET /api/options returns (used by the React form and the chat)."""
    fields = {f: [{"value": k, "label": v["label"]} for k, v in opts.items()] for f, opts in FIELDS.items()}
    fields["properties"] = property_options(classes)
    fields["function"] = function_options()
    return {
        "fields": fields,
        "design_functions": {k: {"classes": v["classes"], "load_mode": v["load_mode"],
                                 "index": v["index"]["text"], "free_variable": v["free_variable"]}
                             for k, v in FUNCTIONS.items()},
        "multi_select": ["properties"],
        "min_limits": MIN_LIMIT_FIELDS,
        "families": FAMILIES,
        "material_groups": {k: v["label"] for k, v in price_categories.items()},
        "priority_weight": PRIORITY_WEIGHT,
    }
