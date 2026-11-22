from openai.types.chat import ChatCompletionMessageParam

SYSTEM_PROMPT = (
    "Опирайся на исходный текст. Не выдумывай факты и не добавляй сведений, "
    "которых нет в исходном тексте."
)

USER_PROMPT_TEMPLATE = (
    "Проанализируй исходный текст и представь результат в трёх разделах "
    "в следующем порядке:\n"
    "1. Краткое содержание.\n"
    "2. Ключевые мысли: три ключевые мысли текста.\n"
    "3. Итоговый ответ: полезный ответ по содержанию текста.\n\n"
    "Исходный текст:\n{text}"
)


def build_prompt(text: str) -> str:
    """Сформировать инструкцию для обработки текста моделью."""
    return USER_PROMPT_TEMPLATE.format(text=text)


def build_messages(text: str) -> list[ChatCompletionMessageParam]:
    """Сформировать сообщения с общими правилами и задачей для текста."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(text)},
    ]
