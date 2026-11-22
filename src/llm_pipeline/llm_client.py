import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def get_response(messages: list[ChatCompletionMessageParam]) -> str:
    """Отправить готовые сообщения модели и вернуть текст ответа."""
    load_dotenv()
    with OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    ) as client:
        response = client.chat.completions.create(
            model=os.environ["LLM_MODEL"],
            messages=messages,
        )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Модель не вернула текст ответа")
    return content
