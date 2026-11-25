import unittest

from pydantic import ValidationError

from llm_pipeline.schemas import ClassificationResult


class ClassificationSchemaTest(unittest.TestCase):
    def test_accepts_each_category_with_an_intent(self) -> None:
        for category in ("support", "feedback", "complaint", "sales", "general_question"):
            with self.subTest(category=category):
                result = ClassificationResult.model_validate(
                    {"category": category, "intent": "Узнать, как настроить сервис"}
                )
                self.assertEqual(result.category, category)
                self.assertEqual(result.intent, "Узнать, как настроить сервис")

    def test_rejects_invalid_classifications(self) -> None:
        cases: tuple[tuple[dict[str, object], str], ...] = (
            ({"category": "unknown", "intent": "Узнать условия"}, "category"),
            ({"intent": "Узнать условия"}, "category"),
            ({"category": "sales"}, "intent"),
            ({"category": "sales", "intent": ""}, "intent"),
            ({"category": "sales", "intent": " \t\n"}, "intent"),
            ({"category": "sales", "intent": 42}, "intent"),
            ({"category": "sales", "intent": None}, "intent"),
        )
        for payload, field in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError) as raised:
                    ClassificationResult.model_validate(payload)
                self.assertEqual(raised.exception.errors()[0]["loc"], (field,))
