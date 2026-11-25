import io
import json
import runpy
import unittest
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI
from pydantic import TypeAdapter
from typing_extensions import TypedDict


class _StubResponse(TypedDict):
    summary: str
    category: str
    sentiment: str
    key_points: list[str]
    final_answer: str


class _RequestMessage(TypedDict):
    role: str
    content: str


class _RequestBody(TypedDict):
    messages: list[_RequestMessage]


_REQUEST_BODY: TypeAdapter[_RequestBody] = TypeAdapter(_RequestBody)


ROOT = Path(__file__).resolve().parents[1]
RESPONSES: dict[str, _StubResponse] = {
    "kolobok.txt": {
        "summary": "Колобок убегает от деда и бабки, но попадается лисе.",
        "category": "Народная сказка",
        "sentiment": "Шутливая с печальным финалом",
        "key_points": [
            "Бабка печёт колобок из остатков муки.",
            "Колобок поёт зверям песенку и убегает.",
            "Лиса просит спеть поближе и съедает его.",
        ],
        "final_answer": "Хитрость лисы срабатывает там, где прямые угрозы не помогли.",
    },
    "3pigs.txt": {
        "summary": "Три брата строят разные дома, но от волка их спасает каменный дом.",
        "category": "Литературная сказка",
        "sentiment": "Тревожная с радостным финалом",
        "key_points": [
            "Наф-Наф заранее готовится к холодам.",
            "Волк разрушает дома двух братьев.",
            "Все трое укрываются у Наф-Нафа и остаются жить вместе.",
        ],
        "final_answer": "Сказка показывает, зачем готовиться заранее и помогать друг другу.",
    },
    "pismo_tatiane.txt": {
        "summary": "Татьяна признаётся адресату в любви и ждёт его ответа.",
        "category": "Стихотворное письмо",
        "sentiment": "Тревожная и взволнованная",
        "key_points": [
            "Сначала Татьяна хотела скрыть свои чувства.",
            "Она видит во встрече с адресатом знак судьбы.",
            "Татьяна боится укора, но доверяет его чести.",
        ],
        "final_answer": "В письме надежда на взаимность соседствует со страхом быть отвергнутой.",
    },
    "support_request.txt": {
        "summary": "Пользовательница не может скачать октябрьский счёт и просит выслать его на почту.",
        "category": "Обращение в поддержку",
        "sentiment": "Нейтральная",
        "key_points": [
            "Счёт не открывается в личном кабинете.",
            "Повторная попытка скачивания не помогла.",
            "Пользовательница хочет получить счёт по почте.",
        ],
        "final_answer": "Проверьте ошибку {invoice_unavailable} и помогите отправить октябрьский счёт на почту.",
    },
    "product_review.txt": {
        "summary": "Термокружка сохраняет тепло, но её крышка протекает на боку.",
        "category": "Отзыв о товаре",
        "sentiment": "Смешанная",
        "key_points": [
            "Напиток остаётся тёплым до обеда.",
            "Кружку удобно держать в руке.",
            "В сумке крышка подтекает, если кружка лежит на боку.",
        ],
        "final_answer": "Кружка подходит для стола, а в сумке её лучше держать вертикально.",
    },
}


class Day3ExamplesTest(unittest.TestCase):
    def test_cli_renders_all_fields_for_each_example(self) -> None:
        for filename, response in RESPONSES.items():
            with self.subTest(filename=filename):
                self.check_example(filename, response)

    def check_example(self, filename: str, response: _StubResponse) -> None:
        path = ROOT / filename
        requests: list[httpx.Request] = []

        def reply(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "role": "assistant",
                    "content": json.dumps(response, ensure_ascii=False),
                }}],
            })

        settings = {
            "DEEPSEEK_API_KEY": "local-test-key",
            "LLM_BASE_URL": "https://example.invalid/v1",
            "LLM_MODEL": "local-test-model",
        }
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            path.open(encoding="utf-8") as stdin,
            httpx.Client(transport=httpx.MockTransport(reply), trust_env=False) as http,
            patch.dict("os.environ", settings),
            patch("llm_pipeline.llm_client.load_dotenv"),
            patch("llm_pipeline.llm_client.OpenAI", partial(OpenAI, http_client=http)),
            patch("sys.stdin", stdin),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exit_state,
        ):
            runpy.run_module("llm_pipeline.main", run_name="__main__", alter_sys=True)

        self.assertEqual(exit_state.exception.code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(len(requests), 1)
        body = _REQUEST_BODY.validate_json(requests[0].content, strict=True)
        user_message, = (message for message in body["messages"] if message["role"] == "user")
        _, separator, sent_text = user_message["content"].partition("Исходный текст:\n")
        self.assertTrue(separator)
        self.assertEqual(sent_text, path.read_text(encoding="utf-8"))

        first, second, third = response["key_points"]
        self.assertEqual(
            stdout.getvalue(),
            f"Краткое содержание: {response['summary']}\n"
            f"Категория: {response['category']}\n"
            f"Тональность: {response['sentiment']}\n\n"
            "Ключевые мысли:\n"
            f"1. {first}\n2. {second}\n3. {third}\n\n"
            f"Итоговый ответ:\n{response['final_answer']}\n",
        )
