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

CLASSIFICATION_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n\nОпредели категорию запроса и кратко опиши, чего хочет автор.\n"
    "- support: помощь с проблемой или настройкой.\n"
    "- feedback: отзыв или предложение без требования исправить конкретную ситуацию.\n"
    "- complaint: претензия с ожиданием исправления или компенсации.\n"
    "- sales: интерес к покупке, ценам или условиям.\n"
    "- general_question: остальные вопросы и тексты без явного запроса.\n"
    "Если в тексте несколько целей, выбирай категорию по главной цели автора, "
    "а не по тональности. Если запроса нет, укажи это в intent, не приписывая "
    "автору цель.\n\n"
    "Верни ровно один JSON-объект с двумя строковыми полями category и intent, "
    "без Markdown-обёртки и пояснений до или после объекта. Пример:\n"
    '{"category": "support", "intent": "Узнать, как сбросить пароль"}'
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


def build_classification_messages(text: str) -> list[ChatCompletionMessageParam]:
    """Сформировать сообщения для классификации, не изменяя исходный текст."""
    return [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
