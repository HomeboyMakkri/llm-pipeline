import re
from typing import get_args

from pydantic import TypeAdapter

from llm_pipeline.prompts import build_generation_messages
from llm_pipeline.schemas import Category, ClassificationResult


def test_category_changes_only_the_route_instruction() -> None:
    source = "  Не получается оформить заказ.\nЧто означает {ошибка}?\n\n"
    intent = "Узнать, почему не получается оформить заказ"
    expected_cues: dict[Category, tuple[str, ...]] = {
        "support": ("структурированные шаги",),
        "feedback": ("Поблагодари", "суть"),
        "complaint": ("неудобство", "без необоснованных обещаний"),
        "sales": ("опиши пользу", "не выдумывай цены и условия"),
        "general_question": ("Ответь прямо", "попроси уточнение"),
    }
    assert set(expected_cues) == set(get_args(Category))

    common_prompts: set[str] = set()
    for category, cues in expected_cues.items():
        classification = ClassificationResult(category=category, intent=intent)
        system, user = build_generation_messages(source, classification)
        assert system["role"] == "system"
        assert user == {
            "role": "user",
            "content": f"Намерение автора:\n{intent}\n\nИсходный текст:\n{source}",
        }
        assert isinstance(system["content"], str)
        common, separator, route = system["content"].rpartition("\n\n")
        assert separator
        assert intent not in system["content"]
        assert source not in system["content"]
        for other_category, other_cues in expected_cues.items():
            assert other_category not in system["content"]
            for cue in other_cues:
                assert (cue in route) == (cue in cues)
                assert cue not in common
        common_prompts.add(common)

    assert len(common_prompts) == 1


def test_generation_example_has_only_the_four_output_fields() -> None:
    classification = ClassificationResult(category="support", intent="Сбросить пароль")
    system, _ = build_generation_messages("Как сбросить пароль?", classification)
    assert system["role"] == "system"
    prompt = system["content"]
    assert isinstance(prompt, str)
    example = re.search(r"\{[^{}]*\}", prompt)
    assert example is not None
    payload = TypeAdapter(dict[str, object]).validate_json(example.group())
    assert payload.keys() == {"summary", "sentiment", "key_points", "final_answer"}
    for field in ("summary", "sentiment", "final_answer"):
        assert isinstance(payload[field], str)
    key_points = TypeAdapter(list[str]).validate_python(payload["key_points"], strict=True)
    assert len(key_points) == 3
