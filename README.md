# PDFScanner

PDFScanner is a local document-processing application that extracts text from
PDFs, images, and Word documents. It includes a Flask web interface and a
command-line interface, with optional OpenAI-powered document classification
and summarization.

## Features

- Extract embedded text from PDFs
- Run PaddleOCR on scanned PDFs and images
- Extract text from DOCX files
- Process one file or an entire directory from the CLI
- Classify and summarize documents with OpenAI
- Download explanations as PDF and web results as JSON

## Supported Files

- PDF
- PNG
- JPG and JPEG
- DOCX

## Requirements

- Python 3.12
- PaddlePaddle and PaddleOCR
- An OpenAI API key only when using classification and summarization

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

For OpenAI classification and summarization, create a `.env` file:

```env
OPENAI_API_KEY=your_api_key_here
```

Basic text extraction does not require an OpenAI API key.

## Web Application

Start the Flask application:

```cmd
python web_app.py
```

Open `http://127.0.0.1:5000` in a browser, select a document, and choose either:

- **Explain:** generate a plain-English explanation and optionally translate it
- **Extract:** OCR and text extraction, with optional classification and summarization

Uploaded files are deleted after processing. JSON downloads are generated in
the browser from the current result and are not retained by the web server.

The document explainer is available at `/`, and the text extractor is available
at `/extract`. The explainer uses `gpt-5.4-mini` by default. Override it with:

```env
OPENAI_EXPLAIN_MODEL=your_model_name
```

Production safety limits can be adjusted with environment variables:

```env
MAX_PDF_PAGES=10
OPENAI_RATE_LIMIT_SHORT=3 per hour
OPENAI_RATE_LIMIT_DAILY=10 per day
RATE_LIMIT_STORAGE_URI=memory://
MAX_CONCURRENT_JOBS=1
FULL_ANALYSIS_ENABLED=true
TURNSTILE_ENABLED=true
TURNSTILE_SITE_KEY=your_turnstile_site_key
TURNSTILE_SECRET_KEY=your_turnstile_secret_key
```

Use a Redis storage URI such as `redis://localhost:6379/0` for rate limiting
when running multiple workers or containers. `MAX_CONCURRENT_JOBS` limits
simultaneous OCR/OpenAI jobs per running Python process.
`FULL_ANALYSIS_ENABLED=false` disables `/api/extract` Analyze mode while keeping
plain text extraction available. Leave Turnstile disabled for local-only testing
if you do not have keys yet, but set `TURNSTILE_ENABLED=true` with Cloudflare
Turnstile keys before public launch.

## Command Line

Extract text from one document:

```cmd
python -m agent data\samples\invoice_test.png
```

Extract, classify, and summarize:

```cmd
python -m agent data\samples\invoice_test.png --mode full
```

You can also provide a directory to process every file inside it:

```cmd
python -m agent data\samples
```

CLI results are saved to `output/results.json` and
`output/extracted_text.txt`.

## Docker

Build and run the configured container:

```cmd
docker compose build
docker compose run --rm pdfscanner data/samples/invoice_test.png
```

Use `--mode full` to enable OpenAI classification and summarization.

## PaddleOCR-VL Experiment

An isolated PaddleOCR-VL test is available at `experiments/paddleocr_vl_extract.py`.
It uses the local `PaddleOCR-VL-0.9B` model with `PP-DocLayoutV2` for basic text extraction.
Run it with `python experiments\paddleocr_vl_extract.py <document-path>`.
Its output is saved under `output/paddleocr_vl/`.
This experiment does not modify or run through the Flask application or the primary PP-OCRv5 pipeline.

## Project Structure

```text
agent/                  Core extraction, classification, and summarization
data/samples/           Sample documents
experiments/            Isolated model experiments
static/                 Web interface assets
templates/              Flask HTML templates
tests/                  Automated tests
web_app.py              Flask application entry point
```

## Notes

- The first PaddleOCR run may download model files and take longer to start.
- OCR processing time depends on document size and available CPU/GPU resources.
- Files in `output/`, `.env`, and local virtual environments are ignored by Git.
