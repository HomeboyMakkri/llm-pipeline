import sys

from .llm_client import get_response
from .prompts import build_messages


def main() -> None:
    text = sys.stdin.read()
    messages = build_messages(text)
    response = get_response(messages)
    print(response)


if __name__ == "__main__":
    main()
