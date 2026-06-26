import os
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


DEFAULT_MODEL = os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-4o-mini")


class DocumentClassification(BaseModel):
    document_type: Literal[
        "invoice",
        "resume",
        "contract",
        "letter",
        "bank_statement",
        "other",
    ]
    confidence: int = Field(ge=0, le=100)


def _client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def classify_document(text: str) -> dict:
    response = _client().responses.parse(
        model=DEFAULT_MODEL,
        instructions=(
            "Classify the document into exactly one of these categories: "
            "invoice, resume, contract, letter, bank_statement, or other. "
            "Use only the supplied document text. Return a confidence score "
            "from 0 to 100."
        ),
        input=text[:3000],
        text_format=DocumentClassification,
    )

    if response.output_parsed is None:
        raise ValueError("The classification model did not return a usable result.")

    return response.output_parsed.model_dump()
