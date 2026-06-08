import json
import os

from openai import OpenAI
from pydantic import BaseModel, Field


DEFAULT_MODEL = os.getenv("OPENAI_EXPLAIN_MODEL", "gpt-5.4-mini")


class DocumentExplanation(BaseModel):
    document_type: str
    summary: str
    explanation: str
    important_points: list[str] = Field(default_factory=list)
    actions_required: list[str] = Field(default_factory=list)
    important_dates: list[str] = Field(default_factory=list)
    amounts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def explain_document(text: str) -> dict:
    response = _client().responses.parse(
        model=DEFAULT_MODEL,
        reasoning={"effort": "low"},
        instructions=(
            "Explain the provided English document in clear, plain English. "
            "Use only facts found in the document. Preserve names, dates, amounts, "
            "addresses, account numbers, and legal references exactly. Identify "
            "required actions and meaningful risks, but do not provide professional "
            "legal, medical, or financial advice. Use empty lists when a category "
            "does not apply."
        ),
        input=text,
        text_format=DocumentExplanation,
    )

    if response.output_parsed is None:
        raise ValueError("The explanation model did not return a usable result.")

    return response.output_parsed.model_dump()


def translate_explanation(explanation: dict, target_language: str) -> dict:
    response = _client().responses.parse(
        model=DEFAULT_MODEL,
        reasoning={"effort": "low"},
        instructions=(
            f"Translate every user-facing value in the supplied JSON into "
            f"{target_language}. Preserve the JSON structure and keys. Do not add, "
            "remove, summarize, or reinterpret information. Preserve names, dates, "
            "amounts, addresses, account numbers, and legal references exactly."
        ),
        input=json.dumps(explanation, ensure_ascii=False),
        text_format=DocumentExplanation,
    )

    if response.output_parsed is None:
        raise ValueError("The translation model did not return a usable result.")

    return response.output_parsed.model_dump()
