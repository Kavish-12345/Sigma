"""
Engineering fit: how well a material's real properties suit each application class.

The trained model answers almost yes or no for each class, so on its own it cannot tell a
good candidate from a better one. This adds a graded 0-1 score from the property that
matters for that class. The ranges are plain engineering rules of thumb, chosen by hand,
and are not fitted to data. No LLM is involved here.
"""
from __future__ import annotations

import pandas as pd

ML_WEIGHT = 0.6   # suitability = 60% trained model + 40% engineering fit
HB_PER_SU = 1 / 3.45  # Brinell hardness estimated as Su/3.45 where the dataset has no hardness
PROVEN_SCORE = 0.7    # lowest suitability for a class the material's own data proves
HARDENED = r"quench|temper|harden|heat treat|improved|nitrid|carburi"  # "Improved" = Q+T in GOST data

CAST_GROUPS = ("aluminium_cast", "cast_steel", "stainless_cast", "grey_cast_iron", "ductile_iron")

# How naturally a material group suits a class that one property cannot describe
HARDENING = {"alloy_steel": 1.0, "spring_steel": 1.0, "carbon_steel": 0.75, "cast_steel": 0.7,
             "stainless_ferritic": 0.6, "aluminium_high_strength": 0.6, "beryllium_copper": 0.6,
             "free_cutting_steel": 0.5, "ductile_iron": 0.5, "heat_resistant_steel": 0.4, "aluminium_cast": 0.4}
BEARING = {"bronze": 0.75, "beryllium_copper": 0.7, "grey_cast_iron": 0.65, "ductile_iron": 0.6}
HOT_STRENGTH = {"heat_resistant_steel": 1.0, "stainless_austenitic": 0.8, "stainless_mo": 0.8, "stainless_ferritic": 0.6,
                "alloy_steel": 0.4, "cast_steel": 0.3, "ductile_iron": 0.3, "grey_cast_iron": 0.3, "carbon_steel": 0.2}
CASTING = {**{g: 1.0 for g in CAST_GROUPS}, "bronze": 0.5, "brass": 0.5, "magnesium_alloy": 0.4, "magnesium_re": 0.4}
MACHINING = {"free_cutting_steel": 1.0, "brass": 0.8, "magnesium_alloy": 0.7, "grey_cast_iron": 0.6,
             "aluminium_wrought": 0.5, "aluminium_high_strength": 0.5, "carbon_steel": 0.4}


# Normal engineering practice, for ranking only: nothing here removes a material.
# Normal groups rank first, then any group not listed, then unusual groups.
STAINLESS = ("stainless_austenitic", "stainless_ferritic", "stainless_mo", "stainless_cast")
COPPER = ("brass", "bronze", "copper_nickel", "copper_pure", "beryllium_copper")
LIGHT = ("aluminium_cast", "aluminium_wrought", "aluminium_high_strength", "magnesium_alloy", "magnesium_re")
IRON = ("grey_cast_iron", "ductile_iron")
USAGE = {
    "gear": {"normal": ("alloy_steel",),
             "unusual": ("free_cutting_steel", "spring_steel", "heat_resistant_steel", *STAINLESS, *LIGHT)},
    "shaft": {"normal": ("alloy_steel", "carbon_steel"),
              "unusual": ("spring_steel", "aluminium_cast", "magnesium_alloy", "magnesium_re", *IRON, *COPPER)},
    "rod": {"normal": ("alloy_steel",),
            "unusual": ("free_cutting_steel", "spring_steel", "grey_cast_iron", *STAINLESS, *COPPER)},
    "spring": {"normal": ("spring_steel",),
               "unusual": ("cast_steel", "free_cutting_steel", *IRON, *LIGHT)},
    "vessel": {"normal": ("carbon_steel", "alloy_steel", "stainless_mo", "stainless_austenitic"),
               "unusual": ("free_cutting_steel", "spring_steel", *IRON, "magnesium_alloy", "magnesium_re")},
    "beam": {"normal": ("carbon_steel", "aluminium_wrought", "aluminium_high_strength"),
             "unusual": ("spring_steel", "free_cutting_steel", "grey_cast_iron")},
}
PART_TYPE_OF_FUNCTION = {"gear": "gear", "shaft": "shaft", "tie": "rod", "spring": "spring",
                         "pressure_vessel": "vessel", "beam": "beam"}


def usage_rank(group: pd.Series, part_type: str | None) -> pd.Series:
    """2 = normal for this kind of part, 1 = no rule, 0 = unusual. No known part type: all equal."""
    rule = USAGE.get(part_type, {})
    rank = {**{g: 2 for g in rule.get("normal", ())}, **{g: 0 for g in rule.get("unusual", ())}}
    return group.map(rank).fillna(1).astype(int)


def _ramp(x: pd.Series, low: float, high: float) -> pd.Series:
    """0 at low, 1 at high, straight line between."""
    return ((x - low) / (high - low)).clip(0, 1)


def proven_by_data(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Rows whose own data proves the class, whatever the model says: hardened means heat-treatable,
    and a wrought steel with real yield strength and measured ductility can carry structural load."""
    text = df["Heat treatment"].fillna("").str.lower()
    hardened = (text.str.contains(HARDENED) & ~text.str.contains("anneal")) | (df["ht_cat"] == "solution_aged")
    wrought_steel = (df["family"] == "ferrous") & ~df["price_category"].isin(CAST_GROUPS + ("free_cutting_steel",))
    return {"Heat-treatable": hardened & (df["price_category"].map(HARDENING).fillna(0) >= 0.5),
            "Structural": wrought_steel & (df["Sy"] >= 235) & (df["A5"] >= 10) & ~df["A5_imputed"]}


def class_fit(df: pd.DataFrame) -> dict[str, pd.Series]:
    """0-1 score per class. Needs the cleaned table plus price_category and corrosion_score."""
    group = df["price_category"]

    def by_group(table: dict, default: float) -> pd.Series:
        return group.map(table).fillna(default).astype(float)

    hardness = df["Bhn"].fillna(df["Su"] * HB_PER_SU)
    stored_energy = df["Sy"] ** 2 / (2 * df["E"])  # MJ/m3: what a spring can store before it yields
    return {
        "Structural": 0.6 * _ramp(df["Sy"], 200, 800) + 0.4 * _ramp(df["A5"], 5, 25),
        "Heat-treatable": 0.6 * by_group(HARDENING, 0.1) + 0.4 * _ramp(df["Su"], 500, 1300),
        "Wear-resistant": pd.concat([_ramp(hardness, 150, 500), by_group(BEARING, 0.0)], axis=1).max(axis=1),
        "Spring": _ramp(stored_energy, 0.3, 4.5),
        "Corrosion-resistant": (df["corrosion_score"] - 1) / 4,
        "Heat-resistant": by_group(HOT_STRENGTH, 0.05),
        "Castings": by_group(CASTING, 0.05).where(df["ht_cat"] != "as_cast", 1.0),
        "Sheet forming": 0.7 * _ramp(df["A5"], 10, 40) + 0.3 * (1 - _ramp(df["Sy_Su"], 0.5, 0.9)),
        "Free-machining": by_group(MACHINING, 0.2),
    }
