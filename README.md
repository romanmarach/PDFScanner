# PDFScanner

PDFScanner is a full-stack document intelligence application built with Python
and Flask. It extracts text from PDFs, images, and DOCX files using native
parsing plus PaddleOCR fallback for scanned content.

The app includes a web UI and CLI, optional OpenAI-powered document
classification, summarization, plain-English explanation, and translation,
Dockerized deployment, Redis-backed rate limiting, secure upload validation,
automatic cleanup, and a 100+ test suite with GitHub Actions CI.

## Key Capabilities

- Extract embedded text from PDFs with OCR fallback for scanned or low-text pages
- Run PaddleOCR on PNG, JPG, JPEG, and scanned PDF content
- Extract DOCX text, including paragraphs, tables, headers, and footers
- Explain documents in plain English and optionally translate explanations
- Classify and summarize documents with OpenAI structured outputs
- Process one file or an entire directory from the CLI
- Return structured JSON from Flask API endpoints
- Enforce upload type, file size, PDF page, image pixel, and DOCX expansion limits
- Protect expensive routes with Redis-backed rate limiting and concurrency limits
- Support optional Cloudflare Turnstile bot protection
- Apply security headers and delete uploaded files after processing
- Run as a non-root user in Docker

## Tech Stack

Python 3.12, Flask, PaddleOCR, PaddlePaddle, OpenAI API, Redis,
Flask-Limiter, Docker, Docker Compose, Gunicorn, PyMuPDF, pdfplumber,
python-docx, Pillow, pytest, and GitHub Actions.

## Architecture

```text
Browser / CLI
      |
      v
Flask app / agent CLI
      |
      v
Upload validation + rate limiting + concurrency guard
      |
      v
Document extraction
  |-- PDF embedded text via pdfplumber
  |-- OCR fallback for scanned PDF pages via PyMuPDF + PaddleOCR
  |-- Image OCR via PaddleOCR
  `-- DOCX parsing via python-docx
      |
      v
Optional OpenAI analysis
  |-- Classification
  |-- Summarization
  |-- Plain-English explanation
  `-- Translation
      |
      v
Structured JSON response / CLI output files
```

## Supported Files

- PDF
- PNG
- JPG and JPEG
- DOCX

## Testing

The project includes 100+ automated tests covering extraction routing,
PaddleOCR result parsing, PDF OCR fallback, DOCX parsing, OpenAI structured
outputs, CLI behavior, Flask API endpoints, upload validation, resource limits,
rate limiting, concurrency protection, Cloudflare Turnstile handling, security
headers, error responses, and upload cleanup.

GitHub Actions runs the unit test suite on pushes and pull requests. The
workflow also builds the Docker image to catch container regressions.

```cmd
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

Integration tests that use real PaddleOCR models and sample files are local-only:

```cmd
python -m pytest tests/ --run-integration -v
```

## Requirements

- Python 3.12
- PaddlePaddle and PaddleOCR
- Docker Desktop, when running the containerized web stack
- Redis, when running the web app locally without Docker Compose
- An OpenAI API key only when using classification, summarization, explanation,
  or translation

## Installation

```bash
git clone <your-repository-url>
cd PdfScanner
python -m venv .venv
```

Activate the virtual environment:

```cmd
.venv\Scripts\activate
```

Install the dependencies:

```cmd
pip install -r requirements.txt
```

For OpenAI-powered features, create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

Basic text extraction does not require an OpenAI API key. Do not commit `.env`
files or API keys.

## Web Application

### Docker

The Docker stack runs the Flask web app and Redis together:

```cmd
docker compose up --build
```

Open `http://127.0.0.1:5000` in a browser.

Inside Docker, the app uses `RATE_LIMIT_STORAGE_URI=redis://redis:6379/0`.
That Redis instance is shared by the web app and persists rate-limit data in the
`redis-data` Docker volume. PaddleOCR model files are cached in the
`paddlex-cache` Docker volume so container recreation does not redownload them
after the first successful model fetch.

The web and CLI containers run as the non-root `appuser`. Writable runtime paths
are limited to uploads, output, and the PaddleOCR cache.

### Local Python

Rate limiting uses Redis. For local development without containerizing the web
app, start Redis before Flask:

```cmd
docker compose up -d redis
```

The default `RATE_LIMIT_STORAGE_URI` is `redis://localhost:6379/0`, which also
works in `.env` when running `python web_app.py` on your machine.

Start the Flask application:

```cmd
python web_app.py
```

Open `http://127.0.0.1:5000` in a browser. The document explainer is available
at `/`, and the text extractor is available at `/extract`.

The web app exposes:

- `POST /api/explain` for plain-English explanation and optional translation
- `POST /api/extract` for OCR/text extraction, with optional classification and
  summarization
- `GET /healthz` for health checks

Uploaded files are deleted after processing. JSON downloads are generated in the
browser from the current result and are not retained by the web server.

## Command Line

Extract text from one document:

```cmd
python -m agent data\samples\invoice_test.png
```

Extract, classify, and summarize:

```cmd
python -m agent data\samples\invoice_test.png --mode full
```

Process every supported file in a directory:

```cmd
python -m agent data\samples
```

CLI results are saved to `output/results.json` and
`output/extracted_text.txt`.

To run the CLI through Docker:

```cmd
docker compose run --rm cli data/samples/invoice_test.png
```

Use `--mode full` to enable OpenAI classification and summarization.

## Security and Privacy

- Supported extensions are centralized in `agent/text_extraction.py`
- Flask enforces a 25 MB upload limit by default
- PDFs are checked for readability and page count before processing
- Images are checked for valid image data and maximum pixel count
- DOCX files are checked for corrupt ZIP data and uncompressed size expansion
- Expensive API routes are rate-limited and guarded by a processing semaphore
- Optional Cloudflare Turnstile verification can be enabled with environment
  variables
- Responses include security headers such as CSP, `X-Frame-Options`, and
  `X-Content-Type-Options`
- Uploaded files are removed after each request, including failure paths
- The Docker image runs the application as a non-root user

## Deployment Configuration

The Docker Compose file binds Redis to `127.0.0.1:6379` and the web app to
`127.0.0.1:5000` on the host. Neither service is published on all host network
interfaces by default.

By default, `TRUSTED_PROXY_COUNT=0`, so forwarded client-IP headers are ignored.
When deploying behind exactly one trusted reverse proxy, such as Cloudflare or
nginx, set `TRUSTED_PROXY_COUNT=1` and set Gunicorn's `FORWARDED_ALLOW_IPS` to
the proxy IP range, or to `*` only when the app port is reachable exclusively
through that proxy.

Useful environment variables:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_FEATURES_ENABLED=true
OPENAI_ANALYSIS_MODEL=gpt-4o-mini
OPENAI_EXPLAIN_MODEL=gpt-5.4-mini
RATE_LIMIT_STORAGE_URI=redis://localhost:6379/0
EXTRACT_RATE_LIMIT=30 per hour
OPENAI_RATE_LIMIT_SHORT=3 per hour
OPENAI_RATE_LIMIT_DAILY=10 per day
MAX_CONCURRENT_JOBS=1
MAX_PDF_PAGES=10
MAX_IMAGE_PIXELS=50000000
MAX_DOCX_UNCOMPRESSED=104857600
TRUSTED_PROXY_COUNT=0
TURNSTILE_ENABLED=false
TURNSTILE_SITE_KEY=
TURNSTILE_SECRET_KEY=
TURNSTILE_TIMEOUT_SECONDS=5
```

## Project Structure

```text
agent/                  Core extraction, classification, explanation, and summarization
static/                 Web interface JavaScript and CSS
templates/              Flask HTML templates
tests/                  Automated unit and local integration tests
web_app.py              Flask application entry point
docker-compose.yml      Local Redis, web, and CLI services
Dockerfile              Production-style container image
```

## Notes

- The first PaddleOCR run may download model files and take longer to start.
- OCR processing time depends on document size and available CPU/GPU resources.
- Files in `output/`, `uploads/`, `.env`, and local virtual environments are
  ignored by Git.
