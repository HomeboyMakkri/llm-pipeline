import json
import sys
from typing import NoReturn

from pydantic import ValidationError

from .llm_client import get_response
from .prompts import build_messages
from .schemas import ModelResponse


def _reject_json_constant(value: str) -> NoReturn:
    # json.loads по умолчанию допускает NaN и Infinity.
    raise ValueError(f"Недопустимая константа JSON: {value}")


def main() -> int:
    text = sys.stdin.read()
    messages = build_messages(text)
    response = get_response(messages)

    try:
        data = json.loads(response, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError):
        print("Не удалось разобрать ответ модели как JSON.", file=sys.stderr)
        return 1

    try:
        result = ModelResponse.model_validate(data)
    except ValidationError as error:
        location = error.errors(include_input=False)[0]["loc"]
        message = "Ответ модели нарушает контракт."
        if location:
            message += f" Проверьте поле «{location[0]}»."
        print(message, file=sys.stderr)
        return 1

    print(f"Краткое содержание: {result.summary}")
    print(f"Категория: {result.category}")
    print(f"Тональность: {result.sentiment}")
    print("\nКлючевые мысли:")
    for number, point in enumerate(result.key_points, start=1):
        print(f"{number}. {point}")
    print(f"\nИтоговый ответ:\n{result.final_answer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
