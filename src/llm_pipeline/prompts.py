from openai.types.chat import ChatCompletionMessageParam

SYSTEM_PROMPT = (
    "Опирайся на исходный текст. Не выдумывай факты и не добавляй сведений, "
    "которых нет в исходном тексте."
)

USER_PROMPT_TEMPLATE = (
    "Проанализируй исходный текст и верни ровно один JSON-объект "
    "без Markdown-обёртки и пояснений до или после объекта. "
    "Используй все пять полей в форме ниже и заполни их по исходному тексту. "
    "В summary используй не более 60 слов, в key_points помести ровно три "
    "ключевые мысли, а в final_answer дай полезный ответ по содержанию "
    "в одном или двух предложениях.\n\n"
    "{{\n"
    '  "summary": "Краткое содержание",\n'
    '  "category": "Краткая метка типа текста или запроса",\n'
    '  "sentiment": "Метка тональности текста",\n'
    '  "key_points": ["Первая ключевая мысль", "Вторая ключевая мысль", '
    '"Третья ключевая мысль"],\n'
    '  "final_answer": "Короткий полезный ответ по содержанию"\n'
    "}}\n\n"
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
