import os

from dotenv import load_dotenv
from openai import OpenAI


def get_response(prompt: str) -> str:
    """Отправить готовый промпт модели и вернуть текст ответа."""
    load_dotenv()
    with OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    ) as client:
        response = client.chat.completions.create(
            model=os.environ["LLM_MODEL"],
            messages=[{"role": "user", "content": prompt}],
        )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Модель не вернула текст ответа")
    return content
