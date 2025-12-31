from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_document(text: str):
    prompt = f"""
    Summarize the document in two ways:

    1. A 2–3 sentence summary.
    2. A bullet-point list.

    Return ONLY valid JSON like:
    {{
        "short_summary": "...",
        "bullet_points": ["...", "..."]
    }}

    Text:
    {text[:4000]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
