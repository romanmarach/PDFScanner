from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def classify_document(text: str):
    prompt = f"""
    You are a document classification AI.

    Classify the following text into one of the categories:
    - invoice
    - resume
    - contract
    - letter
    - bank_statement
    - other

    Return ONLY valid JSON with:
    {{
        "document_type": "...",
        "confidence": 0-100
    }}

    Text:
    {text[:3000]} 
    """
    #truncated your text to no longer than 3000 characters
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content
