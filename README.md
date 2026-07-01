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
- Docker Desktop, when running the containerized web stack
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

Open `http://127.0.0.1:5000` in a browser, select a document, and choose either:

- **Explain:** generate a plain-English explanation and optionally translate it
- **Extract:** OCR and text extraction, with optional classification and summarization

Uploaded files are deleted after processing. JSON downloads are generated in
the browser from the current result and are not retained by the web server.

The document explainer is available at `/`, and the text extractor is available
at `/extract`.

Runtime configuration is handled through environment variables.

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

To run the CLI through Docker:

```cmd
docker compose run --rm cli data/samples/invoice_test.png
```

Use `--mode full` to enable OpenAI classification and summarization.

## Project Structure

```text
agent/                  Core extraction, classification, and summarization
static/                 Web interface assets
templates/              Flask HTML templates
tests/                  Automated tests
web_app.py              Flask application entry point
```

## Notes

- The first PaddleOCR run may download model files and take longer to start.
- OCR processing time depends on document size and available CPU/GPU resources.
- Files in `output/`, `uploads/`, `.env`, and local virtual environments are ignored by Git.

