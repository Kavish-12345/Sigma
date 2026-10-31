"""
Price layer: maps every dataset row to a price category and computes cost.

Prices live in backend/data/prices.json (static, dated, sourced). The user can
override any category price per request.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

PRICES_PATH = Path(__file__).resolve().parents[1] / "data" / "prices.json"


def load_prices(path: Path = PRICES_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Row -> price category
# --------------------------------------------------------------------------
_PREFIXES = re.compile(r"\b(steel|gost|din|en|nf|bs|jis|csn|iso|ansi|sae|aisi)\b")


def _is_alloy_steel(name: str) -> bool:
    n = name.lower()
    m = re.search(r"\b(?:sae|aisi)\s*(\d)(\d)\d\d\b", n)
    if m:  # 13xx = manganese steels
        return m.group(1) in "23456789" or m.group(1, 2) == ("1", "3")
    if re.search(r"\bbs\s*cr\d", n):  # BS 1449 cold-rolled carbon strip, not chromium
        return False
    if "gost 4543" in n or "smn" in n:  # GOST alloy structural steels; JIS SMn / SMnC
        return True
    if re.search(r"\bnf\s*\d{2}[cmnad]+\d", n):  # French codes: C = Cr, M = Mn, N = Ni, D = Mo, A = Al
        return True
    m = re.search(r"\bcsn\s*1([3-6])\d{3}", n)
    if m:
        return True
    m = re.search(r"\bbs\s*([5-9])\d\d[a-z]\d", n)
    if m:
        return True
    core = _PREFIXES.sub(" ", n)
    core = re.sub(r"\d{3,4}-\d{2}", " ", core)  # GOST standard numbers
    return bool(re.search(r"(cr|mo|ni|ch|ncd|\dcd|scm|snc|\dv)", core))


def _stainless_kind(name: str) -> str:
    n = name.lower().replace(" ", "")
    if re.search(r"316|317|nimo|17346|17347|17348|17349|z\d+cnd", n):  # French D = Mo
        return "stainless_mo"
    if re.search(r"x(17|20)crni1[67]", n):  # 431 type: martensitic despite the nickel
        return "stainless_ferritic"
    if re.search(r"(sus|aisi|sae|bs|^)4[0-4]\d|x\d+cr\d+(?!ni)$|x\d+cr1[37]$|1702\d|1704\d"
                 r"|sae51\d{3}|z\d+c\d|z\d+ca\d|z\d+cdmv|x\d+cral|x\d+crmov", n) and "ni" not in n:
        return "stainless_ferritic"  # SAE 514xx/515xx = 400 series; French C = Cr with no N = no nickel
    return "stainless_austenitic"


# "Heat-resistant" in the data also covers boiler and creep steels. They cost and rust like carbon
# or low-alloy steel, so only the high-chromium grades belong in the heat-resistant price group.
_BOILER_CARBON = re.compile(r"p\d{3}gh|\bdin hi+\b|grade (360|430)|\ba\d\dcp\b|\bcsn 11(368|418)|17mn4")
_CREEP_LOW_ALLOY = re.compile(r"\d+crmo\d|1[56]mo3|15d3|15cd405|\bcsn 1(3030|5020|5121|5313)"
                              r"|12ch1mf|12mch|20ch1m1f1|25ch1mf|18ch3mv|20ch3mvf")
_HEAT_RESISTANT_MO = re.compile(r"316s|x2crnimon")  # 316L and duplex grades


def _heat_resistant_kind(name: str) -> str:
    n = name.lower()
    if _HEAT_RESISTANT_MO.search(n):
        return "stainless_mo"
    if _BOILER_CARBON.search(n):
        return "carbon_steel"
    return "alloy_steel" if _CREEP_LOW_ALLOY.search(n) else "heat_resistant_steel"


def _copper_category(name: str, desc: str) -> str:
    n = name.lower()
    m = re.search(r"\bc(\d{5})\b", n)
    if m:
        u = int(m.group(1))
        if 17000 <= u < 18000 or 82000 <= u < 83000:
            return "beryllium_copper"
        if u < 20000 or 80000 <= u < 82000:
            return "copper_pure"
        if u < 50000 or 83000 <= u < 86000 or 87200 <= u < 87300:
            return "brass"
        if 70000 <= u < 80000 or 96000 <= u < 98000:
            return "copper_nickel"
        return "bronze"
    text = f"{n} {desc.lower()}"
    if "brass" in text or re.search(r"\b4232\d\d", n):
        return "brass"
    return "bronze"


def price_category(row: pd.Series) -> str:
    fam = row["family"]
    name = str(row["Material"])
    desc = "" if pd.isna(row.get("Desc")) else str(row["Desc"])
    labels = str(row.get("app_classes", ""))
    n, d = name.lower(), desc.lower()

    if fam == "aluminium":
        grp = str(row.get("group", n))
        if re.search(r"\d{3}\.\d", grp) or ("csn" in grp and "Castings" in labels):
            return "aluminium_cast"
        if re.search(r"\b[27]\d{3}\b", grp) or "alcu" in d:  # 2xxx by composition where the name is a CSN number
            return "aluminium_high_strength"
        return "aluminium_wrought"
    if fam == "copper":
        return _copper_category(name, desc)
    if fam == "magnesium":
        code = n.replace("magnesium alloy", "").strip()
        return "magnesium_re" if re.match(r"(we|ez|qe|ze|hk|hm|hz|qh|zh|ek)", code) else "magnesium_alloy"

    # ---- ferrous ----
    text = f"{n} {d}"
    if "cast iron" in text or re.search(r"\bggg|\bgg\s*-?\s*\d|\bgt[swb]", n):
        if any(k in text for k in ("nodular", "malleable", "ductile", "ggg", "gts", "gtw", "gtb")):
            return "ductile_iron"
        return "grey_cast_iron"
    if "cast steel" in d:
        return "stainless_cast" if "corrosion" in d else "cast_steel"
    if "heat-resist" in d or "heat resist" in d or "high-temperature" in d:
        return _heat_resistant_kind(name)
    if "stainless" in d or "corrosion" in d or re.search(r"\bsus\d|\bsae\s*30\d{3}|\bx\d+crni", n):
        return _stainless_kind(name)
    if "spring" in d:
        return "spring_steel"
    if "free-cutting" in d or "machinability" in d or re.search(r"\b(?:sae|aisi)\s*1[12]\d\d\b|\bsum\d", n):
        return "free_cutting_steel"
    return "alloy_steel" if _is_alloy_steel(name) else "carbon_steel"


# --------------------------------------------------------------------------
# Price lookup and cost
# --------------------------------------------------------------------------
def effective_prices(prices: dict, overrides: dict | None = None) -> dict[str, dict]:
    """
    Category -> {low, high, typical, available, custom, source, ...}.
    typical = midpoint of the cited range (used for ranking only).
    A user price replaces the range with one exact value for this request only.
    """
    out = {}
    for key, c in prices["categories"].items():
        low, high = c.get("low"), c.get("high")
        available = low is not None and high is not None
        out[key] = {
            "low": float(low) if available else None,
            "high": float(high) if available else None,
            "typical": (float(low) + float(high)) / 2 if available else None,
            "available": available,
            "custom": False,
            "label": c["label"],
            "basis": c["basis"],
            "source": c["source"],
            "url": c.get("url", ""),
            "why": c.get("why", ""),
            "date": prices["retrieved_on"],
        }
    for key, value in (overrides or {}).items():
        if key in out and value is not None and float(value) > 0:
            v = float(value)
            out[key].update(low=v, high=v, typical=v, available=True, custom=True, basis="custom",
                            source="Your price (entered by you)", url="")
    return out


def cost_value(price_rs_kg, density_kg_m3: float, volume_cm3: float | None):
    """
    Rs per part = price (Rs/kg) x density (kg/m3) x volume (cm3) x 1e-6.
    Without a volume, returns Rs per cm3 (same formula with volume = 1 cm3),
    which compares materials fairly because it accounts for density.
    """
    if price_rs_kg is None:
        return None
    return price_rs_kg * density_kg_m3 * (volume_cm3 or 1.0) * 1e-6
