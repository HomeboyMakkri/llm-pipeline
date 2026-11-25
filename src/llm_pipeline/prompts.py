from openai.types.chat import ChatCompletionMessageParam

from .schemas import Category, ClassificationResult

SYSTEM_PROMPT = (
    "Опирайся на исходный текст. Не выдумывай факты и не добавляй сведений, "
    "которых нет в исходном тексте."
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

ROUTE_INSTRUCTIONS: dict[Category, str] = {
    "support": "Дай структурированные шаги для решения проблемы или настройки.",
    "feedback": "Поблагодари за отзыв и отрази его суть.",
    "complaint": (
        "Признай неудобство и предложи следующий шаг без необоснованных обещаний."
    ),
    "sales": "Кратко опиши пользу и следующий шаг, не выдумывай цены и условия.",
    "general_question": "Ответь прямо, а если явного вопроса нет, попроси уточнение.",
}

GENERATION_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n\nИспользуй переданное намерение. Не классифицируй текст повторно "
    "и не определяй намерение заново.\n"
    "Верни ровно один JSON-объект без Markdown-обёртки и пояснений вокруг него. "
    "Используй только четыре поля из примера. В summary используй не более 60 слов, "
    "в key_points помести ровно три ключевые мысли, а в final_answer дай краткий "
    "ответ по инструкции ниже.\n"
    "{\n"
    '  "summary": "Краткое содержание",\n'
    '  "sentiment": "Метка тональности текста",\n'
    '  "key_points": ["Первая ключевая мысль", "Вторая ключевая мысль", '
    '"Третья ключевая мысль"],\n'
    '  "final_answer": "Короткий полезный ответ"\n'
    "}"
)


def build_classification_messages(text: str) -> list[ChatCompletionMessageParam]:
    """Сформировать сообщения для классификации, не изменяя исходный текст."""
    return [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def build_generation_messages(
    text: str, classification: ClassificationResult
) -> list[ChatCompletionMessageParam]:
    """Сформировать сообщения с инструкцией для уже проверенной категории."""
    instruction = ROUTE_INSTRUCTIONS[classification.category]
    return [
        {
            "role": "system",
            "content": f"{GENERATION_SYSTEM_PROMPT}\n\nИнструкция для final_answer: {instruction}",
        },
        {
            "role": "user",
            "content": f"Намерение автора:\n{classification.intent}\n\nИсходный текст:\n{text}",
        },
    ]
