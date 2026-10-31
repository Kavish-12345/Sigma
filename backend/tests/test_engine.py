"""Run:  python -m pytest backend/tests -q"""
import numpy as np
import pytest

from backend.engine.recommend import load_engine, recommend
from backend.engine.topsis import topsis

BASE = dict(properties=["Structural"], environment="dry_indoor",
            weight_priority="important", cost_priority="important")


def test_topsis_simple():
    # A dominates B on both criteria -> A scores higher
    m = np.array([[0.9, 100.0], [0.5, 200.0]])
    s = topsis(m, np.array([1, 1]), np.array([True, False]))
    assert s[0] > s[1] and s[0] == pytest.approx(1.0)


def test_every_row_has_price_category():
    df = load_engine()["df"]
    assert df["price_category"].notna().all()
    assert len(df) == 1552 - 26  # EXCLUDE rows removed


def test_recommend_returns_ranked_results():
    res = recommend(BASE)
    r = res["results"]
    assert len(r) == 10
    assert [x["rank"] for x in r] == list(range(1, 11))
    scores = [x["topsis_score"] for x in r]
    assert scores == sorted(scores, reverse=True)
    assert len({x["material"] for x in r}) == 10  # one entry per alloy


@pytest.mark.parametrize("material, group", [
    ("NF 30CND8", "alloy_steel"), ("NF 42C4", "alloy_steel"), ("NF 16NC6", "alloy_steel"), ("NF 40CAD612", "alloy_steel"),
    ("Steel 35G2 GOST 4543-71", "alloy_steel"), ("Steel SAE 1340", "alloy_steel"), ("JIS SMn433", "alloy_steel"),
    ("BS CR2PL", "carbon_steel"), ("EN P265GH", "carbon_steel"), ("DIN HII", "carbon_steel"), ("NF XC10", "carbon_steel"),
    ("EN 13CrMo4", "alloy_steel"), ("EN 16Mo3", "alloy_steel"), ("Steel 25Ch1MF GOST 20072-74", "alloy_steel"),
    ("Steel SAE 51410", "stainless_ferritic"), ("Steel SAE 51440C", "stainless_ferritic"), ("NF Z20C13", "stainless_ferritic"),
    ("EN X17CrNi16-2", "stainless_ferritic"), ("DIN X6CrAl13", "stainless_ferritic"),
    ("NF Z2CND1712", "stainless_mo"), ("BS 316S11", "stainless_mo"), ("EN X2CrNiMoN22-5-3", "stainless_mo"),
    ("JIS SUS304", "stainless_austenitic"), ("NF Z2CN1810", "stainless_austenitic"), ("Steel SAE 30304", "stainless_austenitic"),
    ("EN X6CrNiTi18-10", "heat_resistant_steel"), ("Steel SAE 1095", "carbon_steel"), ("Steel SAE 4140", "alloy_steel"),
    ("CSN 423045", "bronze"), ("CSN 423213", "brass"), ("CSN 424201", "aluminium_high_strength"),
    ("CSN 424413", "aluminium_wrought"),
])
def test_material_is_in_the_right_price_group(material, group):
    df = load_engine()["df"]
    assert set(df.loc[df["Material"] == material, "price_category"]) == {group}


def test_soft_copper_scores_below_spring_steel_as_a_spring():
    r = recommend({**BASE, "properties": ["Spring"], "top_n": 50})["results"]
    steel = [x["suitability_pct"] for x in r if x["material_group"] == "spring_steel"]
    brass = [x["suitability_pct"] for x in r if x["material_group"] == "brass"]
    assert steel and (not brass or max(brass) < min(steel))


def test_suitability_is_graded():
    r = recommend({**BASE, "properties": ["Structural"], "top_n": 30})["results"]
    assert len({round(x["suitability_pct"]) for x in r}) >= 8


def test_high_temperature_removes_light_alloys():
    r = recommend({**BASE, "environment": "high_temperature", "weight_priority": "critical", "top_n": 50})
    assert r["results"] and not {x["family"] for x in r["results"]} & {"aluminium", "magnesium"}
    assert "Heat-resistant" in r["targets"]
    only_al = recommend({**BASE, "environment": "high_temperature", "include": ["aluminium"]})
    assert only_al["results"] == [] and only_al["problem_fields"] == ["environment"]


def test_hardened_condition_proves_heat_treatable():
    df = load_engine()["df"]
    row = df[(df["Material"] == "Steel 50 GOST 1050-88") & (df["Heat treatment"] == "Improved")]
    assert row["p::Heat-treatable"].iloc[0] >= 0.7
    soft = df[(df["Material"] == "Steel SAE 1095") & (df["Heat treatment"] == "annealed")]
    assert soft["p::Heat-treatable"].iloc[0] < row["p::Heat-treatable"].iloc[0]
    assert df.loc[df["Material"] == "JIS SUS304", "p::Heat-treatable"].max() < 0.5  # austenitic: cannot be hardened


def test_measured_data_proves_structural():
    df = load_engine()["df"]
    assert df.loc[df["Material"] == "EN 42CrMo4", "p::Structural"].max() >= 0.7
    assert df.loc[df["Material"] == "Steel 40Ch GOST 4543-71", "A5_imputed"].any()  # guessed values earn no proof
    weak = df[(df["Material"] == "Grey cast iron class 20")]
    assert weak.empty or weak["p::Structural"].max() < 0.7


GEAR = dict(function="gear", design_inputs={"P_kW": 5, "N_rpm": 1440, "m_mm": 3, "z": 20, "b_mm": 30},
            properties=["Heat-treatable", "Wear-resistant"], primary_property="Wear-resistant",
            cost_priority="important", weight_priority="not_important")


def test_normal_materials_rank_first_and_nothing_is_removed():
    r = recommend({**GEAR, "top_n": 5})["results"]
    assert r[0]["material_group"] == "alloy_steel" and r[0]["usage"] == "Common for this part"
    assert not any(x["material_group"] == "free_cutting_steel" for x in r)
    asked = recommend({**GEAR, "include": ["free_cutting_steel"], "top_n": 5})["results"]
    assert asked and all(x["usage"] == "Unusual for this part" for x in asked)


def test_general_part_uses_its_part_type():
    form = dict(function="general", design_inputs={"section": "rectangle", "b_mm": 18, "h_mm": 30, "F_kN": 25, "M_Nm": 40},
                properties=["Structural"], cost_priority="important", top_n=5)
    assert recommend({**form, "part_type": "rod"})["results"][0]["material_group"] == "alloy_steel"
    assert all(x["usage"] == "" for x in recommend(form)["results"])


def test_results_are_varied():
    r = recommend(BASE)["results"]
    groups = [x["material_group"] for x in r]
    assert len(r) == 10 and len(set(groups)) >= 3
    assert all(groups.count(g) <= 3 for g in groups)
    for i, a in enumerate(r):
        for b in r[:i]:
            pa, pb = a["properties"], b["properties"]
            assert not (a["material_group"] == b["material_group"] and abs(pa["Sy"] - pb["Sy"]) <= 0.1 * pa["Sy"]
                        and abs(pa["Su"] - pb["Su"]) <= 0.1 * pa["Su"])


def test_variety_does_not_bring_in_poor_fits():
    form = dict(function="gear", design_inputs={"P_kW": 5, "N_rpm": 1440, "m_mm": 3, "z": 20, "b_mm": 30},
                properties=["Wear-resistant", "Heat-treatable"], primary_property="Heat-treatable",
                environment="dry_indoor", cost_priority="important", weight_priority="not_important")
    assert all(x["suitability_pct"] >= 50 for x in recommend(form)["results"])
    spring = recommend(dict(properties=["Spring"]))["results"]  # better to repeat a group than pad with weak fits
    assert all(x["material_group"] == "spring_steel" for x in spring)


def test_short_list_still_fills_up():  # one group only: variety must not cost results
    r = recommend({**BASE, "include": ["carbon_steel"]})["results"]
    assert len(r) == 10 and {x["material_group"] for x in r} == {"carbon_steel"}


def test_same_form_same_answer():  # No-AI form and chat must give identical results
    assert recommend(BASE)["results"] == recommend(dict(BASE))["results"]


def test_limits_and_avoid():
    res = recommend({**BASE, "min_limits": {"Sy": 600}, "avoid": ["ferrous"]})
    for x in res["results"]:
        assert x["properties"]["Sy"] >= 600 and x["family"] != "ferrous"


def test_price_range_and_override():
    a = recommend({**BASE, "volume_cm3": 100, "include": ["carbon_steel"], "top_n": 1})["results"][0]
    assert a["cost"]["low"] < a["cost"]["high"] and not a["price"]["custom"]
    b = recommend({**BASE, "volume_cm3": 100, "include": ["carbon_steel"], "top_n": 1,
                   "price_overrides": {"carbon_steel": 100}})["results"][0]
    assert b["price"]["custom"] and b["cost"]["low"] == b["cost"]["high"]
    assert b["cost"]["low"] == pytest.approx(100 * b["properties"]["Ro"] * 100e-6, rel=1e-3)


def test_table_not_changed_by_override():
    recommend({**BASE, "price_overrides": {"carbon_steel": 999}})
    assert load_engine()["prices"]["categories"]["carbon_steel"]["low"] == 52


def test_bad_option_rejected():
    with pytest.raises(ValueError):
        recommend({**BASE, "environment": "on_mars"})


def test_properties_come_from_model():
    from backend.engine.options import options_payload
    eng = load_engine()
    opts = options_payload(eng["prices"]["categories"], eng["classes"])
    assert [o["value"] for o in opts["fields"]["properties"]] == eng["classes"]


def test_primary_property_weighting_and_not_sure():
    a = recommend({**BASE, "properties": ["Wear-resistant", "Corrosion-resistant"], "primary_property": "Wear-resistant"})
    assert a["targets"] == {"Wear-resistant": 3, "Corrosion-resistant": 2}
    b = recommend({**BASE, "properties": ["not_sure"]})
    assert b["targets"] == {} and len(b["results"]) == 10   # best fit over all classes
