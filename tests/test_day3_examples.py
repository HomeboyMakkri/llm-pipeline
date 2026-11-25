import io
import json
import runpy
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import httpx
from openai import OpenAI
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from llm_pipeline.schemas import Category


class _StubResponse(TypedDict):
    summary: str
    category: Category
    intent: str
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
        "category": "general_question",
        "intent": "Явного запроса нет: автор рассказывает сказку.",
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
        "category": "general_question",
        "intent": "Явного запроса нет: автор рассказывает о трёх поросятах.",
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
        "category": "general_question",
        "intent": "Явного запроса к читателю нет: автор передаёт признание Татьяны.",
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
        "category": "support",
        "intent": "Получить октябрьский счёт по почте.",
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
        "category": "feedback",
        "intent": "Рассказать о достоинствах и недостатках кружки.",
        "sentiment": "Смешанная",
        "key_points": [
            "Напиток остаётся тёплым до обеда.",
            "Кружку удобно держать в руке.",
            "В сумке крышка подтекает, если кружка лежит на боку.",
        ],
        "final_answer": "Кружка подходит для стола, а в сумке её лучше держать вертикально.",
    },
}


class RoutedCliTest(unittest.TestCase):
    def test_cli_renders_all_fields_for_each_example(self) -> None:
        for filename, response in RESPONSES.items():
            with self.subTest(filename=filename):
                self.check_example(filename, response)

    def check_example(self, filename: str, response: _StubResponse) -> None:
        text = (ROOT / filename).read_text(encoding="utf-8")
        classification = {"category": response["category"], "intent": response["intent"]}
        generation = {key: value for key, value in response.items() if key not in classification}
        code, stdout, stderr, requests = self.run_cli(
            text,
            [json.dumps(classification, ensure_ascii=False), json.dumps(generation, ensure_ascii=False)],
        )

        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        classification_request, generation_request = requests
        self.assertEqual(classification_request["messages"][1], {"role": "user", "content": text})
        self.assertIn('"category"', classification_request["messages"][0]["content"])
        self.assertIn('"intent"', classification_request["messages"][0]["content"])
        self.assertEqual(generation_request["messages"][1], {
            "role": "user",
            "content": f"Намерение автора:\n{response['intent']}\n\nИсходный текст:\n{text}",
        })
        route_cues: dict[str, str] = {
            "general_question": "попроси уточнение",
            "support": "структурированные шаги",
            "feedback": "Поблагодари",
        }
        self.assertIn(route_cues[response["category"]], generation_request["messages"][0]["content"])

        first, second, third = response["key_points"]
        self.assertEqual(
            stdout,
            f"Краткое содержание: {response['summary']}\n"
            f"Категория: {response['category']}\n"
            f"Намерение: {response['intent']}\n"
            f"Тональность: {response['sentiment']}\n\n"
            "Ключевые мысли:\n"
            f"1. {first}\n2. {second}\n3. {third}\n\n"
            f"Итоговый ответ:\n{response['final_answer']}\n",
        )

    def test_invalid_answers_report_the_stage_without_partial_output(self) -> None:
        classification = '{"category":"support","intent":"Сбросить пароль"}'
        cases = [
            (["{"], "Классификация: не удалось разобрать ответ модели как JSON."),
            (
                ['{"category":"unknown","intent":"Сбросить пароль"}'],
                "Классификация: ответ модели нарушает контракт. Проверьте поле «category».",
            ),
            ([classification, "{"], "Генерация: не удалось разобрать ответ модели как JSON."),
            (
                [
                    classification,
                    '{"summary":"Вопрос","sentiment":"Нейтральная",'
                    '"key_points":["Только одна мысль"],"final_answer":"Откройте настройки."}',
                ],
                "Генерация: ответ модели нарушает контракт. Проверьте поле «key_points».",
            ),
        ]
        for replies, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                code, stdout, stderr, requests = self.run_cli("Как сбросить пароль?", replies)
                self.assertEqual(code, 1)
                self.assertEqual(stdout, "")
                self.assertEqual(stderr, diagnostic + "\n")
                self.assertEqual(len(requests), len(replies))

    def run_cli(
        self, text: str, replies: list[str]
    ) -> tuple[object, str, str, list[_RequestBody]]:
        requests: list[_RequestBody] = []
        remaining = iter(replies)

        def reply(request: httpx.Request) -> httpx.Response:
            requests.append(_REQUEST_BODY.validate_json(request.content, strict=True))
            content = next(remaining)
            return httpx.Response(200, json={
                "choices": [{"message": {
                    "role": "assistant",
                    "content": content,
                }}],
            })

        def client(api_key: str, base_url: str) -> OpenAI:
            return OpenAI(
                api_key=api_key,
                base_url=base_url,
                http_client=httpx.Client(transport=httpx.MockTransport(reply), trust_env=False),
            )

        settings = {
            "DEEPSEEK_API_KEY": "local-test-key",
            "LLM_BASE_URL": "https://example.invalid/v1",
            "LLM_MODEL": "local-test-model",
        }
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch.dict("os.environ", settings),
            patch("llm_pipeline.llm_client.load_dotenv"),
            patch("llm_pipeline.llm_client.OpenAI", client),
            patch("sys.stdin", io.StringIO(text)),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as exit_state,
        ):
            runpy.run_module("llm_pipeline.main", run_name="__main__", alter_sys=True)

        return exit_state.exception.code, stdout.getvalue(), stderr.getvalue(), requests
