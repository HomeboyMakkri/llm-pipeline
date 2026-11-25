import json
import sys
from typing import NoReturn, TypeVar

from pydantic import BaseModel, ValidationError

from .llm_client import get_response
from .prompts import build_classification_messages, build_generation_messages
from .schemas import ClassificationResult, GenerationResult

_Result = TypeVar("_Result", bound=BaseModel)


def _reject_json_constant(value: str) -> NoReturn:
    # json.loads по умолчанию допускает NaN и Infinity.
    raise ValueError(f"Недопустимая константа JSON: {value}")


def _parse_response(response: str, schema: type[_Result], stage: str) -> _Result | None:
    try:
        data = json.loads(response, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError):
        print(f"{stage}: не удалось разобрать ответ модели как JSON.", file=sys.stderr)
        return None

    try:
        return schema.model_validate(data)
    except ValidationError as error:
        location = error.errors(include_input=False)[0]["loc"]
        message = f"{stage}: ответ модели нарушает контракт."
        if location:
            message += f" Проверьте поле «{location[0]}»."
        print(message, file=sys.stderr)
        return None


def main() -> int:
    text = sys.stdin.read()
    classification = _parse_response(
        get_response(build_classification_messages(text)),
        ClassificationResult,
        "Классификация",
    )
    if classification is None:
        return 1

    result = _parse_response(
        get_response(build_generation_messages(text, classification)),
        GenerationResult,
        "Генерация",
    )
    if result is None:
        return 1

    print(f"Краткое содержание: {result.summary}")
    print(f"Категория: {classification.category}")
    print(f"Намерение: {classification.intent}")
    print(f"Тональность: {result.sentiment}")
    print("\nКлючевые мысли:")
    for number, point in enumerate(result.key_points, start=1):
        print(f"{number}. {point}")
    print(f"\nИтоговый ответ:\n{result.final_answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
