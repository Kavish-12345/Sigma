"""Part-aware questions: the LLM's reply is validated by code (no LLM call needed)."""
import pytest

from backend.chat import llm
from backend.chat.form_state import apply_answer, empty_form, normalise_form, to_recommend_payload
from backend.chat.part_detail import (MAX_QUESTIONS, answer_detail, apply_effect, apply_step, clean_effect,
                                      clean_question, detail_spec, hint_fits, needs_details, set_hints)

GROUPS = ["ferrous", "aluminium"]
CLASSES = ["Heat-resistant", "Heat-treatable", "Structural", "Wear-resistant"]
QUESTION = {"done": False, "info": "Rods see reversing inertia loads.", "question": "How hot does it run?",
            "options": [{"label": "Normal engine temperature", "effect": {"loading": "cyclic"}},
                        {"label": "Very hot", "effect": {"environment": "high_temperature", "classes": ["Heat-resistant"],
                                                         "note": "Check strength at temperature."}}]}
FINISH = {"done": True, "summary": "It sees reversing load.", "function": "general",
          "guide": {"properties": "Fatigue matters.", "design_inputs": "Enter the material class.", "made_up": "dropped"}}
HINTS = {"hints": {"F_kN": "typically 10-40 kN", "made_up": "dropped", "section": "dropped: not a number input"}}


def form_with_part():
    f = empty_form()
    f["part_name"] = "connecting rod"
    return f


def test_effect_keeps_only_what_the_engine_understands():
    raw = {"classes": ["Wear-resistant", "Flying"], "environment": "space", "cost_priority": "critical",
           "loading": "sideways", "min_limits": {"Sy": 9000}, "note": "  Check toughness. "}
    assert clean_effect(raw, CLASSES) == {"classes": ["Wear-resistant"], "cost_priority": "critical",
                                          "note": "Check toughness."}
    assert clean_effect("nonsense", CLASSES) == {}


def test_badly_shaped_effect_is_dropped_not_a_crash():
    bad = {"classes": [None, 5, {}], "environment": ["chemical"], "cost_priority": {}, "loading": ["shock"]}
    assert clean_effect(bad, CLASSES) == {}


def test_question_that_changes_nothing_is_rejected():
    useless = {"question": "Favourite colour?", "options": [{"label": "Red"}, {"label": "Blue"}]}
    assert clean_question(useless, CLASSES) is None
    assert clean_question({"question": "Only one?", "options": QUESTION["options"][:1]}, CLASSES) is None
    assert clean_question(QUESTION, CLASSES)["options"][1]["value"] == "1"


def test_priority_needs_a_question_about_it():
    q = {"question": "How hot does it run?", "options": [
        {"label": "Hot", "effect": {"cost_priority": "critical", "weight_priority": "important", "loading": "cyclic"}},
        {"label": "Cool", "effect": {"cost_priority": "important"}}]}
    assert clean_question(q, CLASSES)["options"][0]["effect"] == {"loading": "cyclic"}
    assert clean_question({**q, "question": "Does cost matter more?"}, CLASSES)["options"][1]["effect"]


def test_milder_environment_does_not_replace_a_harsher_one():
    f = form_with_part()
    apply_effect(f, {"environment": "high_temperature"}, GROUPS, CLASSES)
    apply_effect(f, {"environment": "outdoor_humid"}, GROUPS, CLASSES)
    assert f["environment"] == "high_temperature"
    apply_effect(f, {"environment": "chemical"}, GROUPS, CLASSES)
    assert f["environment"] == "chemical"


def test_shock_answer_is_not_downgraded():
    f = form_with_part()
    apply_effect(f, {"loading": "shock"}, GROUPS, CLASSES)
    apply_effect(f, {"loading": "cyclic"}, GROUPS, CLASSES)
    assert f["design_draft"]["loading"] == "shock"


def test_hint_in_the_wrong_unit_is_dropped():
    assert not hint_fits("~30-50 kN peak force", "N")
    assert hint_fits("30000-50000 N", "N") and hint_fits("≈5-20 Nm", "N·m") and hint_fits("20-80 teeth", "")
    assert hint_fits("0.5-2 kN", "kN") and hint_fits("typical for this part", "mm")
    f = form_with_part()
    apply_step(f, {**FINISH, "function": "beam"}, GROUPS, CLASSES)
    set_hints(f, {"hints": {"F_N": "30-50 kN", "L_mm": "120-150 mm"}})
    assert f["hints"] == {"L_mm": "120-150 mm"}
    assert "buckling" not in f["guide"]["design_inputs"]  # a beam card has no such field


@pytest.mark.parametrize("label, clean", [
    ("Medium (20,000-100,000 km or ~50,000 cycles)", "Medium (20,000-100,000 km)"),
    ("~30,000 km (≈1.5 ×10⁶ cycles)", "~30,000 km"),
    ("Long service life", "Long service life"), ("Forged, 2 passes", "Forged, 2 passes"),
])
def test_made_up_cycle_counts_are_removed(label, clean):
    q = clean_question({"question": "How long?", "options": [{"label": label, "effect": {"loading": "cyclic"}},
                                                             {"label": "Other"}]}, CLASSES)
    assert q["options"][0]["label"] == clean


def test_guidance_with_internal_codes_is_dropped():
    f = form_with_part()
    apply_step(f, {**FINISH, "guide": {"properties": "needs d_mm and F_kN"}}, GROUPS, CLASSES)
    assert "properties" not in f["guide"]


def test_answer_applies_its_effect():
    f = form_with_part()
    assert apply_step(f, QUESTION, GROUPS, CLASSES) == QUESTION["info"]
    spec = detail_spec(f)
    assert spec["field"] == "part_detail" and "effect" not in spec["options"][0]
    answer_detail(f, "1", "Very hot", GROUPS, CLASSES)
    assert f["environment"] == "high_temperature" and f["suggested"] == ["Heat-resistant"]
    assert f["notes"] == ["Check strength at temperature."] and f["pending_detail"] is None
    assert to_recommend_payload(f)["part_notes"] == f["notes"]


def test_not_sure_changes_nothing_and_typed_answer_gets_effect_from_next_reply():
    f = form_with_part()
    apply_step(f, QUESTION, GROUPS, CLASSES)
    answer_detail(f, "not_sure", "Not sure", GROUPS, CLASSES)
    assert f["environment"] is None and not f["details"][-1]["typed"]
    apply_step(f, QUESTION, GROUPS, CLASSES)
    answer_detail(f, None, "it runs at about 600 C", GROUPS, CLASSES)
    assert f["details"][-1]["typed"]
    apply_step(f, {**FINISH, "last_answer_effect": {"environment": "high_temperature"}}, GROUPS, CLASSES)
    assert f["environment"] == "high_temperature"


def test_finish_sets_function_keeps_loading_and_filters_guidance():
    f = form_with_part()
    apply_step(f, QUESTION, GROUPS, CLASSES)
    answer_detail(f, "0", "", GROUPS, CLASSES)
    assert apply_step(f, FINISH, GROUPS, CLASSES) == FINISH["summary"]
    assert f["function"] == "general" and f["design_draft"]["loading"] == "cyclic"
    assert f["guide"]["properties"] == "Fatigue matters." and "made_up" not in f["guide"]
    assert "buckling" in f["guide"]["design_inputs"] and "material class" not in f["guide"]["design_inputs"]
    set_hints(f, HINTS)
    assert f["hints"] == {"F_kN": "typically 10-40 kN"}
    assert not needs_details(f)
    assert f["design_inputs"] is None  # hints are guidance: the user still enters every value


def test_card_with_missing_values_continues_without_the_calculation():
    f = form_with_part()
    assert apply_answer(f, "function", "general", "", GROUPS, CLASSES) is None
    card = {"section": "rectangle", "b_mm": 10, "h_mm": 18}
    assert apply_answer(f, "design_inputs", card, "", GROUPS, CLASSES) is None
    assert f["design_inputs"] == "not_sure" and "skipped" in f["notes"][0]
    assert to_recommend_payload(f)["design_inputs"] is None


def test_card_with_a_wrong_value_still_gets_a_message():
    f = form_with_part()
    apply_answer(f, "function", "general", "", GROUPS, CLASSES)
    card = {"section": "hollow_round", "d_mm": 20, "di_mm": 25, "F_kN": 1}
    assert "Inner diameter" in apply_answer(f, "design_inputs", card, "", GROUPS, CLASSES)
    assert f["design_inputs"] is None


def test_part_type_is_kept_only_if_known():
    f = form_with_part()
    apply_step(f, {**FINISH, "part_type": "rod"}, GROUPS, CLASSES)
    assert f["part_type"] == "rod" and to_recommend_payload(f)["part_type"] == "rod"
    g = form_with_part()
    apply_step(g, {**FINISH, "part_type": ["rod"]}, GROUPS, CLASSES)
    assert g["part_type"] is None


def test_blank_part_name_falls_back_to_the_users_words():
    f = empty_form()
    assert apply_answer(f, "part_name", "", "A connecting rod", GROUPS) is None
    assert f["part_name"] == "A connecting rod"
    assert apply_answer(empty_form(), "part_name", "", "", GROUPS) is not None


def test_state_survives_the_browser_round_trip():
    f = form_with_part()
    apply_step(f, QUESTION, GROUPS, CLASSES)
    again = normalise_form(f)
    assert again["pending_detail"] == f["pending_detail"] and needs_details(again)


class _Fake:
    """Stands in for the streaming client: the form-filling pass returns nothing."""
    class chat:
        class completions:
            create = staticmethod(lambda **kw: [])


def _run(monkeypatch, form, replies, direct=None):
    monkeypatch.setattr(llm, "_PART_CACHE", {})
    monkeypatch.setattr(llm, "_client", lambda: _Fake())
    monkeypatch.setattr(llm, "_ask_json", lambda *a: replies.pop(0))
    return llm.chat_turn([{"role": "user", "content": "connecting rod"}], form, {}, direct, CLASSES)


def test_chat_asks_part_question_then_fixed_form(monkeypatch):
    done = _run(monkeypatch, form_with_part(), [QUESTION])
    assert done["question"]["field"] == "part_detail" and done["reply"] == QUESTION["info"]
    done = _run(monkeypatch, done["form"], [FINISH, HINTS], {"field": "part_detail", "value": "1", "label": "Very hot"})
    assert done["form"]["details_done"] and done["question"]["field"] == "design_inputs"
    assert done["question"]["info"].startswith("Typical values")
    hint = next(o["hint"] for o in done["question"]["options"] if o["key"] == "F_kN")
    assert hint == "typically 10-40 kN"


def test_chat_retries_a_rejected_tool_call(monkeypatch):
    attempts = []

    def create(**kw):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("Failed to parse tool call arguments as JSON")
        return []

    class Flaky(_Fake):
        class chat:
            class completions:
                pass
    Flaky.chat.completions.create = staticmethod(create)
    monkeypatch.setattr(llm, "_PART_CACHE", {})
    monkeypatch.setattr(llm, "_client", lambda: Flaky())
    monkeypatch.setattr(llm, "_ask_json", lambda *a: QUESTION)
    done = llm.chat_turn([{"role": "user", "content": "rod"}], form_with_part(), {}, None, CLASSES)
    assert len(attempts) == 2 and done["question"]["field"] == "part_detail"


def test_same_part_and_answers_reuse_the_saved_question(monkeypatch):
    calls = []
    monkeypatch.setattr(llm, "_PART_CACHE", {})
    monkeypatch.setattr(llm, "_client", lambda: _Fake())
    monkeypatch.setattr(llm, "_ask_json", lambda *a: calls.append(1) or QUESTION)
    for _ in range(3):
        llm.chat_turn([{"role": "user", "content": "rod"}], form_with_part(), {}, None, CLASSES)
    assert len(calls) == 1


def test_chat_falls_back_to_fixed_form_when_the_model_fails(monkeypatch):
    done = _run(monkeypatch, form_with_part(), ["not json", None])
    assert done["form"]["details_done"] and done["question"]["field"] == "function"


def test_questions_are_capped(monkeypatch):
    f = form_with_part()
    f["details"] = [{"question": str(i), "answer": "x"} for i in range(MAX_QUESTIONS)]
    done = _run(monkeypatch, f, [QUESTION, QUESTION])  # the model ignores the instruction to finish
    assert done["form"]["details_done"] and not done["form"]["pending_detail"]
