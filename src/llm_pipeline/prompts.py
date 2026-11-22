from openai.types.chat import ChatCompletionMessageParam

SYSTEM_PROMPT = (
    "Опирайся на исходный текст. Не выдумывай факты и не добавляй сведений, "
    "которых нет в исходном тексте."
)

USER_PROMPT_TEMPLATE = (
    "Проанализируй исходный текст: кратко изложи содержание не более чем "
    "в 60 словах, выдели ровно три ключевые мысли в виде нумерованных пунктов "
    "и дай полезный ответ по содержанию в одном или двух предложениях. "
    "Используй следующий каркас ответа:\n\n"
    "Краткое содержание\n\n"
    "Ключевые мысли\n"
    "1.\n"
    "2.\n"
    "3.\n\n"
    "Итоговый ответ\n\n"
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
