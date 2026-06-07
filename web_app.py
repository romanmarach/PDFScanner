import json
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

from agent.text_extraction import extract_text


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".docx"}


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


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
    OUTPUT_DIR.mkdir(exist_ok=True)

    original_name = secure_filename(uploaded_file.filename)
    suffix = Path(original_name).suffix.lower()
    stored_name = f"{uuid.uuid4().hex}{suffix}"
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

        output_file = OUTPUT_DIR / "latest_result.json"
        output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500
    finally:
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.get("/api/latest")
def latest_result():
    output_file = OUTPUT_DIR / "latest_result.json"
    if not output_file.exists():
        return jsonify({"error": "No extraction result has been saved yet."}), 404

    return send_file(output_file, as_attachment=True, download_name="pdfscanner-result.json")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
