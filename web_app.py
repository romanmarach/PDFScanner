import json
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from agent.text_extraction import extract_text


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".docx"}
ALLOWED_LANGUAGES = {
    "english": "English",
    "ukrainian": "Ukrainian",
    "spanish": "Spanish",
    "french": "French",
    "german": "German",
    "portuguese": "Portuguese",
    "polish": "Polish",
    "russian": "Russian",
}
MAX_EXPLAIN_CHARS = 60_000


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def upload_names(filename: str) -> tuple[str, str]:
    """Return a display name and a unique on-disk name for an upload.

    The suffix comes from the raw filename because secure_filename() strips
    non-ASCII characters and can drop the extension entirely (e.g. "файл.pdf").
    """
    suffix = Path(filename).suffix.lower()
    display_name = secure_filename(filename)
    if not Path(display_name).suffix:
        display_name = f"document{suffix}"
    return display_name, f"{uuid.uuid4().hex}{suffix}"


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({"error": "File is larger than the 25 MB upload limit."}), 413


def parse_jsonish(value):
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/extract")
def extractor():
    return render_template("extract.html")


@app.post("/api/extract")
def extract_document():
    uploaded_file = request.files.get("file")
    mode = request.form.get("mode", "extract")

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Choose a file before extracting text."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Supported files are PDF, PNG, JPG, JPEG, and DOCX."}), 400

    if mode not in {"extract", "full"}:
        return jsonify({"error": "Invalid processing mode."}), 400

    UPLOAD_DIR.mkdir(exist_ok=True)

    original_name, stored_name = upload_names(uploaded_file.filename)
    stored_path = UPLOAD_DIR / stored_name
    uploaded_file.save(stored_path)

    try:
        text = extract_text(str(stored_path))

        result = {
            "fileName": original_name,
            "mode": mode,
            "text": text,
            "characterCount": len(text),
            "wordCount": len(text.split()),
        }

        if mode == "full":
            from agent.doc_classify import classify_document
            from agent.doc_summarize import summarize_document

            result["classification"] = parse_jsonish(classify_document(text))
            result["summary"] = parse_jsonish(summarize_document(text))

        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    finally:
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/api/explain")
def explain_uploaded_document():
    uploaded_file = request.files.get("file")
    language_key = request.form.get("language", "english").lower()

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Choose a file before explaining the document."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Supported files are PDF, PNG, JPG, JPEG, and DOCX."}), 400

    if language_key not in ALLOWED_LANGUAGES:
        return jsonify({"error": "Choose a supported explanation language."}), 400

    UPLOAD_DIR.mkdir(exist_ok=True)

    original_name, stored_name = upload_names(uploaded_file.filename)
    stored_path = UPLOAD_DIR / stored_name
    uploaded_file.save(stored_path)

    try:
        from agent.doc_explain import explain_document, translate_explanation

        extracted_text = extract_text(str(stored_path))
        if not extracted_text.strip():
            return jsonify({"error": "No readable text was found in this document."}), 422

        was_truncated = len(extracted_text) > MAX_EXPLAIN_CHARS
        explanation = explain_document(extracted_text[:MAX_EXPLAIN_CHARS])
        language_name = ALLOWED_LANGUAGES[language_key]
        translated = None

        if language_key != "english":
            translated = translate_explanation(explanation, language_name)

        result = {
            "fileName": original_name,
            "language": language_key,
            "languageName": language_name,
            "english": explanation,
            "translated": translated,
            "sourceCharacterCount": len(extracted_text),
            "sourceWasTruncated": was_truncated,
        }

        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    finally:
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
