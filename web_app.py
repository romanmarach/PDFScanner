import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path
from threading import BoundedSemaphore

from flask import Flask, jsonify, render_template, request
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

from agent.text_extraction import EXTRACTABLE_EXTENSIONS, extract_text


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
ALLOWED_EXTENSIONS = EXTRACTABLE_EXTENSIONS
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
GENERIC_PROCESSING_ERROR = "Something went wrong while processing the document. Please try again."
# 'unsafe-inline' in style-src is required by the print/PDF feature, which
# renders an iframe srcdoc containing an inline <style> block that inherits
# this page-level policy.
CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' https://challenges.cloudflare.com",
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
        "font-src 'self' https://fonts.gstatic.com",
        "img-src 'self' data:",
        "connect-src 'self'",
        "frame-src https://challenges.cloudflare.com",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
        "object-src 'none'",
    ]
)
MAX_EXPLAIN_CHARS = 60_000
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_MB = MAX_UPLOAD_BYTES // (1024 * 1024)
MAX_PDF_PAGES = int(os.environ.get("MAX_PDF_PAGES", "10"))
MAX_IMAGE_PIXELS = int(os.environ.get("MAX_IMAGE_PIXELS", str(50_000_000)))
MAX_DOCX_UNCOMPRESSED = int(
    os.environ.get("MAX_DOCX_UNCOMPRESSED", str(100 * 1024 * 1024))
)
OPENAI_RATE_LIMIT_SHORT = os.environ.get("OPENAI_RATE_LIMIT_SHORT", "3 per hour")
OPENAI_RATE_LIMIT_DAILY = os.environ.get("OPENAI_RATE_LIMIT_DAILY", "10 per day")
EXTRACT_RATE_LIMIT = os.environ.get("EXTRACT_RATE_LIMIT", "30 per hour")
RATE_LIMIT_STORAGE_URI = os.environ.get("RATE_LIMIT_STORAGE_URI", "redis://localhost:6379/0")
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "1"))
OPENAI_FEATURES_ENABLED = os.environ.get("OPENAI_FEATURES_ENABLED", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
TURNSTILE_SITE_KEY = os.environ.get("TURNSTILE_SITE_KEY", "")
TURNSTILE_SECRET_KEY = os.environ.get("TURNSTILE_SECRET_KEY", "")
TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
TURNSTILE_TIMEOUT_SECONDS = float(os.environ.get("TURNSTILE_TIMEOUT_SECONDS", "5"))
TRUSTED_PROXY_COUNT = int(os.environ.get("TRUSTED_PROXY_COUNT", "0"))
TURNSTILE_ENABLED_ENV = os.environ.get("TURNSTILE_ENABLED")
if TURNSTILE_ENABLED_ENV is None:
    TURNSTILE_ENABLED = bool(TURNSTILE_SITE_KEY and TURNSTILE_SECRET_KEY)
else:
    TURNSTILE_ENABLED = TURNSTILE_ENABLED_ENV.lower() in {"1", "true", "yes", "on"}


app = Flask(__name__)
if __name__ != "__main__":
    gunicorn_logger = logging.getLogger("gunicorn.error")
    if gunicorn_logger.handlers:
        app.logger.handlers = gunicorn_logger.handlers
        app.logger.setLevel(gunicorn_logger.level)
if TRUSTED_PROXY_COUNT:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=TRUSTED_PROXY_COUNT)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[],
    storage_uri=RATE_LIMIT_STORAGE_URI,
    # If Redis becomes unreachable at runtime, fall back to per-process
    # in-memory limits instead of failing every rate-limited request.
    swallow_errors=True,
    in_memory_fallback_enabled=True,
)
processing_slots = BoundedSemaphore(MAX_CONCURRENT_JOBS)


class PdfValidationError(ValueError):
    """Raised when an uploaded PDF should be rejected before OCR."""


class PdfPageLimitError(PdfValidationError):
    """Raised when an uploaded PDF exceeds the configured page limit."""


class TurnstileValidationError(ValueError):
    """Raised when bot verification fails for a request."""


class TurnstileConfigError(RuntimeError):
    """Raised when bot protection is enabled but not configured."""


def cleanup_upload_dir() -> None:
    if not UPLOAD_DIR.exists():
        return

    for upload_path in UPLOAD_DIR.iterdir():
        try:
            if upload_path.is_file() or upload_path.is_symlink():
                upload_path.unlink()
        except OSError:
            app.logger.warning(
                "Failed to delete stale upload: %s",
                upload_path,
                exc_info=True,
            )


cleanup_upload_dir()

def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def validate_pdf_page_count(path: Path) -> None:
    if path.suffix.lower() != ".pdf":
        return

    import fitz

    read_errors = tuple(
        error_type
        for error_type in (getattr(fitz, "FileDataError", None), RuntimeError)
        if error_type is not None
    )

    try:
        with fitz.open(path) as pdf:
            page_count = pdf.page_count
    except read_errors as exc:
        raise PdfValidationError("This PDF is corrupt or unreadable.") from exc

    if page_count > MAX_PDF_PAGES:
        raise PdfPageLimitError(
            f"PDFs are limited to {MAX_PDF_PAGES} pages. This PDF has {page_count} pages."
        )




def validate_image(path: Path) -> None:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        return

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            width, height = img.size
    except (UnidentifiedImageError, OSError) as exc:
        raise PdfValidationError("This image is corrupt or not a real image.") from exc

    if width * height > MAX_IMAGE_PIXELS:
        raise PdfValidationError("This image's dimensions are too large to process.")


def validate_docx(path: Path) -> None:
    if path.suffix.lower() != ".docx":
        return

    try:
        with zipfile.ZipFile(path) as zf:
            total_uncompressed = sum(info.file_size for info in zf.infolist())
    except zipfile.BadZipFile as exc:
        raise PdfValidationError("This DOCX file is corrupt or unreadable.") from exc

    if total_uncompressed > MAX_DOCX_UNCOMPRESSED:
        raise PdfValidationError("This DOCX expands too large to process.")


def validate_upload_resource_limits(path: Path) -> None:
    validate_pdf_page_count(path)
    validate_image(path)
    validate_docx(path)

def acquire_processing_slot() -> bool:
    return processing_slots.acquire(blocking=False)


def release_processing_slot() -> None:
    processing_slots.release()



def verify_turnstile_response(token: str | None, remote_ip: str | None) -> dict:
    if not TURNSTILE_ENABLED:
        return {"success": True, "skipped": True}

    if not TURNSTILE_SECRET_KEY:
        raise TurnstileConfigError("Bot protection is not configured.")

    if not token:
        raise TurnstileValidationError("Bot verification failed. Please refresh and try again.")

    payload = {
        "secret": TURNSTILE_SECRET_KEY,
        "response": token,
    }
    if remote_ip:
        payload["remoteip"] = remote_ip

    encoded_payload = urllib.parse.urlencode(payload).encode("utf-8")
    verify_request = urllib.request.Request(
        TURNSTILE_VERIFY_URL,
        data=encoded_payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(
            verify_request,
            timeout=TURNSTILE_TIMEOUT_SECONDS,
        ) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise TurnstileValidationError(
            "Bot verification could not be completed. Please try again."
        ) from exc

    if not result.get("success"):
        raise TurnstileValidationError("Bot verification failed. Please refresh and try again.")

    return result


def verify_request_turnstile() -> None:
    verify_turnstile_response(
        request.form.get("cf-turnstile-response"),
        request.remote_addr,
    )


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


@app.after_request
def apply_security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    return response


@app.errorhandler(413)
def upload_too_large(error):
    return jsonify({"error": f"File is larger than the {MAX_UPLOAD_MB} MB upload limit."}), 413


@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({"error": "Too many requests. Please wait before trying again."}), 429


def server_busy_response():
    return jsonify({"error": "Server is busy. Please try again shortly."}), 503


def openai_unavailable_response():
    return jsonify({"error": "AI document features are temporarily unavailable."}), 503


def parse_jsonish(value):
    if not isinstance(value, str):
        return value

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


@app.get("/healthz")
def healthz():
    """Liveness probe for Docker healthchecks and load balancers."""
    return jsonify({"status": "ok"})


@app.get("/")
def index():
    return render_template(
        "index.html",
        max_upload_bytes=MAX_UPLOAD_BYTES,
        max_upload_mb=MAX_UPLOAD_MB,
        turnstile_enabled=TURNSTILE_ENABLED,
        turnstile_site_key=TURNSTILE_SITE_KEY,
    )


@app.get("/extract")
def extractor():
    return render_template(
        "extract.html",
        max_upload_bytes=MAX_UPLOAD_BYTES,
        max_upload_mb=MAX_UPLOAD_MB,
        turnstile_enabled=TURNSTILE_ENABLED,
        turnstile_site_key=TURNSTILE_SITE_KEY,
    )


@app.get("/privacy")
def privacy():
    return render_template("privacy.html")

@app.post("/api/extract")
@limiter.limit(
    EXTRACT_RATE_LIMIT,
    exempt_when=lambda: request.form.get("mode", "extract") != "extract",
)
@limiter.limit(
    OPENAI_RATE_LIMIT_SHORT,
    exempt_when=lambda: request.form.get("mode", "extract") != "full",
)
@limiter.limit(
    OPENAI_RATE_LIMIT_DAILY,
    exempt_when=lambda: request.form.get("mode", "extract") != "full",
)
def extract_document():
    uploaded_file = request.files.get("file")
    mode = request.form.get("mode", "extract")

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Choose a file before extracting text."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Supported files are PDF, PNG, JPG, JPEG, and DOCX."}), 400

    if mode not in {"extract", "full"}:
        return jsonify({"error": "Invalid processing mode."}), 400

    if mode == "full" and not OPENAI_FEATURES_ENABLED:
        return openai_unavailable_response()

    try:
        verify_request_turnstile()
    except TurnstileValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except TurnstileConfigError as exc:
        return jsonify({"error": str(exc)}), 500

    UPLOAD_DIR.mkdir(exist_ok=True)

    original_name, stored_name = upload_names(uploaded_file.filename)
    stored_path = UPLOAD_DIR / stored_name
    uploaded_file.save(stored_path)

    try:
        validate_upload_resource_limits(stored_path)
        if not acquire_processing_slot():
            return server_busy_response()

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

                analysis_errors = {}

                try:
                    result["classification"] = parse_jsonish(classify_document(text))
                except Exception:
                    analysis_errors["classification"] = "Classification could not be completed."

                try:
                    result["summary"] = parse_jsonish(summarize_document(text))
                except Exception:
                    analysis_errors["summary"] = "Summary could not be completed."

                if analysis_errors:
                    result["analysisErrors"] = analysis_errors

            return jsonify(result)
        finally:
            release_processing_slot()
    except PdfValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Extract request failed")
        return jsonify({"error": GENERIC_PROCESSING_ERROR}), 500
    finally:
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            pass


@app.post("/api/explain")
@limiter.limit(OPENAI_RATE_LIMIT_SHORT)
@limiter.limit(OPENAI_RATE_LIMIT_DAILY)
def explain_uploaded_document():
    if not OPENAI_FEATURES_ENABLED:
        return openai_unavailable_response()

    uploaded_file = request.files.get("file")
    language_key = request.form.get("language", "english").lower()

    if uploaded_file is None or uploaded_file.filename == "":
        return jsonify({"error": "Choose a file before explaining the document."}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({"error": "Supported files are PDF, PNG, JPG, JPEG, and DOCX."}), 400

    if language_key not in ALLOWED_LANGUAGES:
        return jsonify({"error": "Choose a supported explanation language."}), 400

    try:
        verify_request_turnstile()
    except TurnstileValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except TurnstileConfigError as exc:
        return jsonify({"error": str(exc)}), 500

    UPLOAD_DIR.mkdir(exist_ok=True)

    original_name, stored_name = upload_names(uploaded_file.filename)
    stored_path = UPLOAD_DIR / stored_name
    uploaded_file.save(stored_path)

    try:
        from agent.doc_explain import explain_document, translate_explanation

        validate_upload_resource_limits(stored_path)
        if not acquire_processing_slot():
            return server_busy_response()

        try:
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
        finally:
            release_processing_slot()
    except PdfValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Explain request failed")
        return jsonify({"error": GENERIC_PROCESSING_ERROR}), 500
    finally:
        try:
            stored_path.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)
