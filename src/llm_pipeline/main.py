import sys

from .llm_client import get_response
from .prompts import build_prompt


def main() -> None:
    text = sys.stdin.read()
    prompt = build_prompt(text)
    response = get_response(prompt)
    print(response)


if __name__ == "__main__":
    main()
