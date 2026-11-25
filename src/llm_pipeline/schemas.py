"""Схемы структурированных ответов модели."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ModelResponse(BaseModel):
    model_config = ConfigDict(strict=True)

    summary: str
    category: str
    sentiment: str
    key_points: list[str] = Field(min_length=3, max_length=3)
    final_answer: str


Category = Literal["support", "feedback", "complaint", "sales", "general_question"]


class ClassificationResult(BaseModel):
    model_config = ConfigDict(strict=True)

    category: Category
    intent: str = Field(pattern=r"\S")
