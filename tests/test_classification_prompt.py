import re
from typing import get_args

from pydantic import TypeAdapter

from llm_pipeline.prompts import build_classification_messages
from llm_pipeline.schemas import Category, ClassificationResult


def test_classification_keeps_source_text_separate_from_instructions() -> None:
    source = "  Не могу войти в аккаунт.\nПодскажите, что значит {ошибка}?\n\n"

    system, user = build_classification_messages(source)

    assert system["role"] == "system"
    assert user["role"] == "user"
    assert user["content"] == source
    assert isinstance(system["content"], str)
    assert source not in system["content"]


def test_classification_instructions_describe_the_schema() -> None:
    system, _ = build_classification_messages("Как сбросить пароль?")
    assert system["role"] == "system"
    instructions = system["content"]
    assert isinstance(instructions, str)
    assert set(get_args(Category)) <= set(re.findall(r"\b\w+\b", instructions))
    assert "JSON" in instructions

    example = re.search(r"\{[^{}]*\}", instructions)
    assert example is not None
    payload = TypeAdapter(dict[str, object]).validate_json(example.group())
    assert payload.keys() == {"category", "intent"}
    ClassificationResult.model_validate(payload)
