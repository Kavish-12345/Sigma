"""Design layer: textbook formulas, screening limits, material index and design check."""
import math

import pytest

from backend.chat.form_state import apply_answer, empty_form
from backend.engine.design import calculate
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
    assert known["requirements"] == {"Sy": 200, "Su": round(2 * 100 / 0.375)}
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
