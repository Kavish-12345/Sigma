"""Chat form validation tests (no LLM call needed). Run: python -m pytest backend/tests -q"""
import time

from backend.chat.form_state import apply_answer, empty_form, missing_fields, to_recommend_payload
from backend.engine.recommend import load_engine, recommend

GROUPS = ["ferrous", "aluminium", "copper", "magnesium", "carbon_steel", "bronze"]
CLASSES = ["Castings", "Corrosion-resistant", "Free-machining", "Heat-resistant", "Heat-treatable",
           "Sheet forming", "Spring", "Structural", "Wear-resistant"]


def test_valid_answers_recorded():
    f = empty_form()
    assert apply_answer(f, "properties", "wear-resistant, Structural", "it slides", GROUPS, CLASSES) is None
    assert f["properties"] == ["Wear-resistant", "Structural"] and f["primary_property"] is None
    assert apply_answer(f, "primary_property", "Structural", "", GROUPS, CLASSES) is None
    assert f["primary_property"] == "Structural"
    apply_answer(f, "suggested_properties", "Spring", "", GROUPS, CLASSES)
    assert f["suggested"] == ["Spring"] and f["properties"] == ["Wear-resistant", "Structural"]
    assert apply_answer(f, "min_limits", "Sy=400, Ro_max=3000", "", GROUPS) is None
    assert f["min_limits"] == {"Sy": 400.0, "Ro_max": 3000.0}
    assert apply_answer(f, "avoid", "none", "", GROUPS) is None and f["avoid"] == []
    assert apply_answer(f, "volume_cm3", "not sure", "", GROUPS) is None and f["volume_cm3"] == "not_sure"


def test_volume_is_entered_in_m3_and_stored_in_cm3():
    for answer, cm3 in (("0.00012", 120), ("1.2e-4", 120), ("1e-7", 0.1), ("120 cm3", 120), ("120,000 mm3", 120)):
        f = empty_form()
        assert apply_answer(f, "volume_cm3", answer, "", GROUPS) is None
        assert abs(f["volume_cm3"] - cm3) < 1e-9


def test_absurd_volume_rejected():
    f = empty_form()
    assert apply_answer(f, "volume_cm3", "1e999", "", GROUPS) is not None and f["volume_cm3"] is None


def test_huge_answer_is_handled_quickly():
    f = empty_form()
    apply_answer(f, "function", "gear", "", GROUPS, CLASSES)
    start = time.time()
    for field in ("design_inputs", "min_limits"):
        apply_answer(f, field, "x" * 100000, "", GROUPS, CLASSES)
    assert time.time() - start < 1


def test_invalid_values_rejected():
    f = empty_form()
    assert apply_answer(f, "environment", "space", "", GROUPS) is not None
    assert apply_answer(f, "properties", "Flying", "", GROUPS, CLASSES) is not None
    assert apply_answer(f, "include", "titanium", "", GROUPS) is not None
    assert f["environment"] is None and f["properties"] is None


def test_complete_and_same_result_as_form():
    f = empty_form()
    answers = {"part_name": "gear shaft", "function": "not_sure", "properties": "Heat-treatable, Structural", "primary_property": "Heat-treatable",
               "environment": "dry_indoor", "weight_priority": "not_important",
               "cost_priority": "important", "volume_cm3": "none", "min_limits": "none",
               "include": "none", "avoid": "none"}
    for k, v in answers.items():
        assert apply_answer(f, k, v, "", GROUPS, CLASSES) is None
    assert missing_fields(f) == []
    chat_payload = to_recommend_payload(f)
    form_payload = dict(properties=["Heat-treatable", "Structural"], primary_property="Heat-treatable",
                        environment="dry_indoor", weight_priority="not_important", cost_priority="important")
    assert recommend(chat_payload)["results"] == recommend(form_payload)["results"]


def test_unit_and_conflict_checks():
    f = empty_form()
    assert "kg/m3" in apply_answer(f, "min_limits", "Ro_max=3", "", GROUPS)
    assert "MPa" in apply_answer(f, "min_limits", "E=200", "", GROUPS)
    assert apply_answer(f, "include", "ferrous", "", GROUPS) is None
    assert apply_answer(f, "avoid", "ferrous", "", GROUPS) is not None  # clash rejected


def test_empty_result_explains_cause():
    res = recommend(dict(properties=["Structural"], include=["ferrous"], avoid=["ferrous"]))
    assert res["results"] == [] and "both" in res["message"]
    res = recommend(dict(properties=["Structural"], min_limits={"Sy": 2900}))
    assert res["results"] == [] and "yield strength" in res["message"]
