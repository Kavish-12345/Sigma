"""
Shared cleaning and feature code.

Used by BOTH the training notebook and the recommendation engine, so the
model sees exactly the same features at training time and at serving time.

Rules kept from the project plan:
  * The professor's property values are never changed in the CSV; cleaning
    always works on a copy.
  * Bhn, HV and pH are NOT model inputs (70-89 % missing).
  * Labels are never derived from the input properties.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Column definitions
# --------------------------------------------------------------------------
NUMERIC_COLS = ["Su", "Sy", "A5", "Bhn", "E", "G", "mu", "Ro", "pH", "HV"]
BASE_FEATURES = ["Su", "Sy", "A5", "E", "mu", "Ro"]
DERIVED_FEATURES = ["Sy_Su"]
NUMERIC_FEATURES = BASE_FEATURES + DERIVED_FEATURES

FAMILIES = ["ferrous", "aluminium", "copper", "magnesium"]
NON_FERROUS = {"aluminium", "aluminum", "copper", "magnesium"}

HT_CATEGORIES = [
    "quenched_tempered",
    "normalized",
    "annealed",
    "solution_aged",
    "cold_worked",
    "hot_worked",
    "as_cast",
    "other",
]

APP_CLASSES = [
    "Structural",
    "Castings",
    "Corrosion-resistant",
    "Heat-treatable",
    "Heat-resistant",
    "Wear-resistant",
    "Free-machining",
    "Spring",
    "Sheet forming",
]

# G values more than this many times the isotropic value E / (2(1+mu))
# are treated as data-entry errors (the stainless rows at ~743,000 MPa).
G_ERROR_RATIO = 2.0


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def to_number(value) -> float:
    """'280 max' -> 280.0, '1,200' -> 1200.0, '' / NaN -> NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # thousands separator
    text = text.replace(",", ".")
    match = _NUM_RE.search(text)
    return float(match.group()) if match else np.nan


def normalise_family(value) -> str:
    """'Aluminium alloy' -> 'aluminium', 'Copper alloy' -> 'copper', 'Ferrous' -> 'ferrous'."""
    v = str(value).strip().lower()
    if "alumin" in v:
        return "aluminium"
    if any(k in v for k in ("copper", "brass", "bronze")):
        return "copper"
    if "magnes" in v:
        return "magnesium"
    if "ferrous" in v and "non" not in v or any(k in v for k in ("steel", "iron")):
        return "ferrous"
    raise ValueError(f"Unknown family value: {value!r}")


def split_labels(value) -> list[str]:
    """'Structural; Spring' -> ['Structural', 'Spring']."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [p.strip() for p in str(value).split(";") if p.strip()]


def label_status(value) -> str:
    """train / exclude / unlabelled / missing, from the app_classes cell."""
    v = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
    up = v.upper()
    if up.startswith("EXCLUDE"):
        return "exclude"
    if up.startswith("UNLABELLED") or up.startswith("UNLABELED"):
        return "unlabelled"
    if not v:
        return "missing"
    return "train"


# --------------------------------------------------------------------------
# Alloy grouping (for grouped cross-validation)
# --------------------------------------------------------------------------
# Temper / condition codes that appear at the END of non-ferrous names:
#   aluminium / magnesium: O, F, W, H14, H321, T6, T651, T7351 ...
#   copper (ASTM B601):    O60, H04, M20, TB00, TD04, TF00, TH04, HR50 ...
_TEMPER_TOKEN = re.compile(
    r"[\s\-_/]+("
    r"O\d*|F|W\d*|M\d+|H\d+\w*|T\d+\w*|"
    r"T[ABDFHLQ]\d+|HR\d+|HT\d*"
    r")\s*$",
    re.IGNORECASE,
)


def alloy_group(material, family: str) -> str:
    """
    Alloy name without the temper suffix, so every temper of one alloy
    lands in the same CV fold.  'Aluminium 6061-T6' and 'Aluminium 6061-O'
    both become 'aluminium 6061'.  'Alclad 2024-T3' joins plain '2024'.

    Ferrous names are left as they are: their heat treatment is in a
    separate column, and stripping tokens such as 'H13' would merge
    different tool steels.
    """
    name = str(material).strip()
    name = re.sub(r"\s+", " ", name)
    name = name.replace("Alclad 202024", "Alclad 2024")  # dataset typo
    if family != "ferrous":
        name = re.sub(r"^alclad\s+", "", name, flags=re.IGNORECASE)
        name = re.sub(r"\balclad\s+", "", name, flags=re.IGNORECASE)
        prev = None
        while prev != name:  # strip repeated suffixes, e.g. '-T6 Alclad'
            prev = name
            name = _TEMPER_TOKEN.sub("", name).strip()
    return name.lower()


# --------------------------------------------------------------------------
# Heat-treatment condition (optional feature)
# --------------------------------------------------------------------------
_HT_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("quenched_tempered", ("quench", "temper", "q+t", "q&t", "q + t", "hardened", "harden",
                           "heat treat", "heat-treat", "improved", "nitrid", "carburi")),  # "Improved" = Q+T in GOST data
    ("normalized", ("normali",)),
    ("solution_aged", ("solution", "aged", "ageing", "aging", "precipitation")),
    ("annealed", ("anneal", "soft", "spheroid")),
    ("cold_worked", ("cold", "drawn", "strain", "work hard", "rolled hard", "half hard")),
    ("as_cast", ("cast",)),
    ("hot_worked", ("hot", "rolled", "forged", "as rolled", "extruded")),
]


def _temper_from_name(material) -> str | None:
    m = _TEMPER_TOKEN.search(str(material))
    if not m:
        return None
    code = m.group(1).upper()
    if code.startswith("T") and not code.startswith(("TD", "TH", "TB")):
        return "solution_aged"
    if code.startswith("O"):
        return "annealed"
    if code.startswith(("H", "TD", "TH")):
        return "cold_worked"
    if code.startswith("M"):
        return "as_cast"
    return None


def ht_category(heat_treatment, material, family: str) -> str:
    text = "" if pd.isna(heat_treatment) else str(heat_treatment).lower()
    for cat, keys in _HT_RULES:
        if any(k in text for k in keys):
            return cat
    if family != "ferrous":
        from_name = _temper_from_name(material)
        if from_name:
            return from_name
    return "other"


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def clean(df_raw: pd.DataFrame, a5_medians: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Returns (clean copy, report).  `a5_medians` lets the engine reuse the
    medians saved at training time instead of recomputing them.
    """
    df = df_raw.copy()
    report: dict = {}

    # 1. text -> numbers ("280 max" -> 280)
    text_counts = {}
    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        original = df[col]
        converted = original.map(to_number)
        is_text = original.notna() & pd.to_numeric(original, errors="coerce").isna()
        text_counts[col] = int(is_text.sum())
        df[col] = converted.astype(float)
    report["text_values_converted"] = {k: v for k, v in text_counts.items() if v}

    # 2. family
    if "family" in df.columns:
        df["family"] = df["family"].map(normalise_family)
    else:
        df["family"] = "ferrous"
    df["is_ferrous"] = df["family"].eq("ferrous")

    # 3. shear modulus: recompute obviously wrong values (stainless ~10x)
    g_calc = df["E"] / (2.0 * (1.0 + df["mu"]))
    bad_g = df["G"].notna() & g_calc.notna() & (df["G"] > G_ERROR_RATIO * g_calc)
    df["G_original"] = df["G"]
    df.loc[bad_g, "G"] = g_calc[bad_g].round(0)
    missing_g = df["G"].isna() & g_calc.notna()
    df.loc[missing_g, "G"] = g_calc[missing_g].round(0)
    df["G_fixed"] = bad_g
    report["G_recomputed_wrong"] = int(bad_g.sum())
    report["G_filled_missing"] = int(missing_g.sum())

    # 4. A5 (elongation): impute median within family
    df["A5_imputed"] = df["A5"].isna()
    if a5_medians is None:
        a5_medians = df.groupby("family")["A5"].median().to_dict()
        a5_medians["_global"] = float(df["A5"].median())
    fill = df["family"].map(a5_medians).fillna(a5_medians["_global"])
    df["A5"] = df["A5"].fillna(fill)
    report["A5_imputed"] = int(df["A5_imputed"].sum())
    report["a5_medians"] = {k: float(v) for k, v in a5_medians.items()}

    # 5. derived feature
    df["Sy_Su"] = df["Sy"] / df["Su"]
    df.loc[~np.isfinite(df["Sy_Su"]), "Sy_Su"] = np.nan
    report["Sy_gt_Su_rows"] = int((df["Sy"] > df["Su"]).sum())

    # 6. heat-treatment category and CV group
    ht_col = "Heat treatment" if "Heat treatment" in df.columns else None
    df["ht_cat"] = [
        ht_category(h if ht_col else None, m, f)
        for h, m, f in zip(df[ht_col] if ht_col else [None] * len(df), df["Material"], df["family"])
    ]
    df["group"] = [alloy_group(m, f) for m, f in zip(df["Material"], df["family"])]

    # 7. label status
    if "app_classes" in df.columns:
        df["status"] = df["app_classes"].map(label_status)
        df["labels"] = [split_labels(v) if s == "train" else [] for v, s in zip(df["app_classes"], df["status"])]
    report["remaining_missing_in_features"] = {
        c: int(df[c].isna().sum()) for c in NUMERIC_FEATURES if df[c].isna().any()
    }
    return df, report


# --------------------------------------------------------------------------
# Feature matrix
# --------------------------------------------------------------------------
def feature_columns(use_family: bool = False, use_ht: bool = False) -> list[str]:
    cols = list(NUMERIC_FEATURES)
    if use_family:
        cols += [f"fam_{f}" for f in FAMILIES]
    if use_ht:
        cols += [f"ht_{h}" for h in HT_CATEGORIES]
    return cols


def build_X(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Build the model input in a fixed column order (same in notebook and engine)."""
    X = pd.DataFrame(index=df.index)
    for col in columns:
        if col.startswith("fam_"):
            X[col] = (df["family"] == col[4:]).astype(float)
        elif col.startswith("ht_"):
            X[col] = (df["ht_cat"] == col[3:]).astype(float)
        else:
            X[col] = df[col].astype(float)
    return X
