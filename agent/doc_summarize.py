import os

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()


DEFAULT_MODEL = os.getenv("OPENAI_ANALYSIS_MODEL", "gpt-4o-mini")


class DocumentSummary(BaseModel):
    short_summary: str
    bullet_points: list[str] = Field(default_factory=list)


def _client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_document(text: str) -> dict:
    response = _client().responses.parse(
        model=DEFAULT_MODEL,
        instructions=(
            "Summarize the supplied document text in two ways: a concise "
            "one-sentence summary and a short bullet-point list. Use only facts "
            "found in the document."
        ),
        input=text[:4000],
        text_format=DocumentSummary,
    )

    if response.output_parsed is None:
        raise ValueError("The summary model did not return a usable result.")

    return response.output_parsed.model_dump()
