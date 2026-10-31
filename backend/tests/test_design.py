"""Design layer: textbook formulas, screening limits, material index and design check."""
import math

import pytest

from backend.chat.form_state import apply_answer, empty_form
from backend.engine.design import calculate, critical_stress
from backend.engine.recommend import recommend

SHAFT = {"P_kW": 10, "N_rpm": 960, "M_Nm": 150, "d_mm": 30, "fos": 2}


def test_shaft_matches_hand_calculation():
    c = calculate("shaft", SHAFT)
    T = 9550 * 10 / 960
    tau = 16 / (math.pi * 30 ** 3) * math.hypot(150, T) * 1000
    assert c["requirements"]["Sy"] == round(2 * 2 * tau)
    assert c["cyclic"]  # rotating bending is fatigue
    assert "Su" in c["requirements"]


def test_gear_lewis():
    c = calculate("gear", {"P_kW": 5, "N_rpm": 1440, "m_mm": 3, "z": 20, "b_mm": 30})
    assert c["cyclic"] and 80 < c["requirements"]["Sy"] / 2 < 100  # sigma ~ 90 MPa


def test_beam_deflection_gives_E_limit():
    c = calculate("beam", {"F_N": 100, "L_mm": 100, "b_mm": 20, "h_mm": 20, "defl_mm": 0.1})
    assert c["requirements"]["E"] > 0


def test_general_combined_loads_match_hand_calculation():
    c = calculate("general", {"section": "solid_round", "d_mm": 20, "F_kN": 10, "M_Nm": 50, "T_Nm": 40})
    A, Z = math.pi * 20 ** 2 / 4, math.pi * 20 ** 3 / 32
    s, t = 10000 / A + 50000 / Z, 40000 / (2 * Z)
    assert c["requirements"]["Sy"] == round(2 * math.sqrt(s ** 2 + 3 * t ** 2))


def test_general_rectangle_torsion_near_exact_square_value():
    c = calculate("general", {"section": "rectangle", "b_mm": 10, "h_mm": 10, "T_Nm": 1, "fos": 1})
    assert c["requirements"]["Sy"] == round(math.sqrt(3) * 4.8 * 1000 / 10 ** 3)  # exact coefficient 4.81


def test_general_tube_is_weaker_than_solid():
    loads = {"d_mm": 30, "M_Nm": 100}
    tube = calculate("general", {**loads, "section": "hollow_round", "di_mm": 20})
    solid = calculate("general", {**loads, "section": "solid_round"})
    assert tube["requirements"]["Sy"] > solid["requirements"]["Sy"]


def test_general_known_stress_and_kt_only_affects_fatigue():
    known = calculate("general", {"section": "known_stress", "stress_MPa": 100, "loading": "cyclic"})
    assert known["requirements"] == {"Sy": 200, "Su": round(2 * 100 / 0.375), "A5": 8}
    plain = {"section": "solid_round", "d_mm": 20, "F_kN": 10, "loading": "cyclic"}
    notched = calculate("general", {**plain, "Kt": 2})
    assert notched["requirements"]["Sy"] == calculate("general", plain)["requirements"]["Sy"]
    assert notched["requirements"]["Su"] > calculate("general", plain)["requirements"]["Su"]


@pytest.mark.parametrize("bad", [
    {"section": "solid_round", "d_mm": 20},                                    # no load
    {"section": "hollow_round", "d_mm": 20, "di_mm": 20, "F_kN": 1},           # no wall
    {"section": "solid_round", "d_mm": 20, "F_kN": 1, "Kt": 0.5},
    {"section": "known_stress"},
])
def test_general_bad_inputs_rejected(bad):
    with pytest.raises(ValueError):
        calculate("general", bad)


def test_general_part_gets_screened_and_checked():
    r = recommend(dict(function="general", design_inputs={"section": "rectangle", "b_mm": 12, "h_mm": 30,
                                                          "F_kN": 15, "M_Nm": 60, "loading": "cyclic"}))
    d = r["design"]
    assert r["results"] and d["function_label"].startswith("General part")
    assert any("First estimate" in s for s in d["steps"])
    assert all(res["properties"]["Sy"] >= d["requirements"]["Sy"] for res in r["results"])


def test_general_chat_keeps_only_the_chosen_section_fields():
    f = empty_form()
    assert apply_answer(f, "function", "General part", "", []) is None
    assert apply_answer(f, "design_inputs", "section=known_stress, stress_MPa=180, d_mm=25", "", []) is None
    assert "d_mm" not in f["design_inputs"] and f["design_inputs"]["stress_MPa"] == 180


@pytest.mark.parametrize("Sy, E, slenderness, expected", [
    (250, 200000, 200, math.pi ** 2 * 200000 / 200 ** 2),                     # long column: Euler
    (250, 200000, 50, 250 * (1 - 50 ** 2 / (2 * 2 * math.pi ** 2 * 200000 / 250))),  # short column: Johnson
])
def test_critical_stress_matches_hand_calculation(Sy, E, slenderness, expected):
    assert float(critical_stress(Sy, E, slenderness)) == pytest.approx(expected)


def test_euler_and_johnson_meet_at_half_yield():
    Sy, E = 300, 200000
    cc = math.sqrt(2 * math.pi ** 2 * E / Sy)
    assert float(critical_stress(Sy, E, cc * 0.9999)) == pytest.approx(Sy / 2, rel=1e-3)
    assert float(critical_stress(Sy, E, cc * 1.0001)) == pytest.approx(Sy / 2, rel=1e-3)


def test_buckling_end_condition_and_section_axis():
    part = {"section": "rectangle", "b_mm": 10, "h_mm": 40, "F_kN": 20, "L_mm": 500}
    pinned, fixed = calculate("general", part), calculate("general", {**part, "ends": "fixed"})
    assert pinned["buckling"]["slenderness"] == pytest.approx(2 * fixed["buckling"]["slenderness"])
    assert pinned["buckling"]["slenderness"] == pytest.approx(500 / math.sqrt(10 ** 2 / 12))  # weak axis governs
    assert calculate("general", {k: v for k, v in part.items() if k != "L_mm"})["buckling"] is None
    assert calculate("tie", {"F_kN": 10, "d_mm": 16, "L_mm": 800})["buckling"]["sigma"] > 0


def test_buckling_screens_and_reports_every_result():
    r = recommend(dict(function="tie", design_inputs={"F_kN": 2, "d_mm": 8, "L_mm": 300}, properties=["Structural"]))
    assert r["results"] and r["design"]["requirements"]["E"] > 0
    for x in r["results"]:
        check = next(c for c in x["design_check"] if c["name"] == "Buckling")
        assert check["ok"] and check["fos"] >= 2
    tall = recommend(dict(function="tie", design_inputs={"F_kN": 50, "d_mm": 5, "L_mm": 2000}, properties=["Structural"]))
    assert not tall["results"] and tall["problem_fields"] == ["design_inputs"]


@pytest.mark.parametrize("fn, bad", [
    ("tie", {"F_kN": "nan", "d_mm": 16}), ("tie", {"F_kN": "inf", "d_mm": 16}), ("tie", {"F_kN": True, "d_mm": 16}),
    ("tie", {"F_kN": 1e300, "d_mm": 16}), ("tie", {"F_kN": 10, "d_mm": 1e-300}),
    ("spring", {"F_N": 5, "Dm_mm": 1, "dw_mm": 1}), ("gear", {"P_kW": 5, "N_rpm": 1440, "m_mm": 3, "z": 5, "b_mm": 30}),
])
def test_impossible_values_get_a_message_not_a_crash(fn, bad):
    with pytest.raises(ValueError):
        calculate(fn, bad)


def test_extra_strength_earns_no_ranking_credit():
    r = recommend(dict(function="tie", design_inputs={"F_kN": 1, "d_mm": 30}, properties=["Structural"],
                       cost_priority="not_important", weight_priority="not_important", top_n=30))
    alike = [x for x in r["results"] if x["properties"]["Ro"] == 7860 and x["material_group"] == "carbon_steel"]
    assert len({x["properties"]["Sy"] for x in alike}) > 1  # different strengths, same price and density...
    assert len({x["index_rel_pct"] for x in alike}) == 1    # ...and the same index once both are strong enough


def test_ductility_limit_by_load_type():
    part = {"section": "solid_round", "d_mm": 20, "F_kN": 10}
    assert "A5" not in calculate("general", part)["requirements"]
    assert calculate("general", {**part, "loading": "cyclic"})["requirements"]["A5"] == 8
    shock = calculate("general", {**part, "loading": "shock"})
    assert shock["requirements"]["A5"] == 11 and shock["cyclic"] and shock["inputs"]["loading"] == "shock"
    assert "A5" not in calculate("gear", {"P_kW": 5, "N_rpm": 1440, "m_mm": 3, "z": 20, "b_mm": 30})["requirements"]


def test_shock_part_gets_ductile_proven_materials():
    r = recommend(dict(function="general", design_inputs={"section": "rectangle", "b_mm": 18, "h_mm": 30, "F_kN": 25,
                                                          "M_Nm": 40, "Kt": 2, "loading": "shock"},
                       properties=["Heat-treatable", "Structural"], primary_property="Heat-treatable"))
    assert r["results"]
    for x in r["results"]:
        assert x["properties"]["A5"] >= 11 and all(c["ok"] for c in x["design_check"])
        assert x["properties"]["Su"] <= 1100 or not x["a5_imputed"]


def test_missing_input_rejected():
    with pytest.raises(ValueError):
        calculate("tie", {"F_kN": 10})


def test_recommend_screens_ranks_and_checks():
    r = recommend(dict(function="shaft", design_inputs=SHAFT, cost_priority="important",
                       weight_priority="not_important"))
    d = r["design"]
    assert d["screening"]["Sy"] == d["requirements"]["Sy"]
    assert "index" in r["weights"] and d["index"]["text"].startswith("σy^(2/3)")
    for res in r["results"]:
        assert res["properties"]["Sy"] >= d["requirements"]["Sy"]
        assert all(c["ok"] for c in res["design_check"]) and res["min_fos"] >= 2


def test_impossible_design_reasks_design_inputs():
    r = recommend(dict(function="tie", design_inputs={"F_kN": 500, "d_mm": 5}))
    assert not r["results"] and r["problem_fields"] == ["design_inputs"]


def test_no_design_still_works():
    r = recommend(dict(properties=["Spring"]))
    assert r["results"] and r["design"] is None


def test_chat_form_design_fields():
    f = empty_form()
    assert apply_answer(f, "function", "gear", "", [], ["Wear-resistant", "Heat-treatable"]) is None
    assert f["suggested"] == ["Wear-resistant", "Heat-treatable"]
    assert apply_answer(f, "design_inputs", "P_kW=5, N_rpm=1440", "", []) is not None  # partial -> kept, asks more
    assert f["design_draft"]["P_kW"] == "5"
    assert apply_answer(f, "design_inputs", {"m_mm": 3, "z": 20, "b_mm": 30}, "", []) is None
    assert f["design_inputs"]["fos"] == 2.0
    f2 = empty_form()
    apply_answer(f2, "function", "not_sure", "", [])
    assert f2["design_inputs"] == "not_sure"
