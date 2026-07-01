import os
# Skip Paddle's model-source network probe during app startup.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
import tempfile

import pdfplumber
import docx
from paddleocr import PaddleOCR

# Source of truth for what extract_text() can handle. The CLI and web app
# derive their accepted-file lists from this; the web app may expose a
# subset, but must never accept an extension that is not in this set.
EXTRACTABLE_EXTENSIONS = frozenset({".pdf", ".png", ".jpg", ".jpeg", ".docx"})
MIN_PAGE_TEXT_CHARS = 20


# PaddleOCR 3.x init (cls/use_angle_cls is replaced by use_textline_orientation)
ocr = PaddleOCR(
    lang="en",
    use_textline_orientation=True,
    # Optional: disable these if you want less overhead
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)


def extract_text(file_path: str) -> str:
    """Main extraction function."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_pdf(file_path)

    elif ext in [".jpg", ".jpeg", ".png"]:
        return ocr_image(file_path)

    elif ext == ".docx":
        return extract_docx(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def extract_pdf(path: str) -> str:
    """Extract each PDF page, using OCR only for pages with little embedded text."""
    with pdfplumber.open(path) as pdf:
        page_texts = [page.extract_text() or "" for page in pdf.pages]

    pages_needing_ocr = [
        page_num
        for page_num, text in enumerate(page_texts)
        if len(text.strip()) < MIN_PAGE_TEXT_CHARS
    ]

    if pages_needing_ocr:
        print(
            f"PDF has {len(pages_needing_ocr)} page(s) with little embedded text; "
            "using OCR for those pages..."
        )
        ocr_texts = ocr_pdf_pages(path, pages_needing_ocr)
        for page_num, ocr_text in ocr_texts.items():
            if ocr_text.strip():
                page_texts[page_num] = ocr_text

    return "\n\n".join(page_texts)


def paddle_predict_to_text(predict_output) -> str:
    """
    Converts PaddleOCR 3.x predict() output (generator of result objects)
    into plain text by extracting recognized text lines.
    """
    lines: list[str] = []

    for res in predict_output:
        j = getattr(res, "json", None)
        if not isinstance(j, dict):
            continue

        core = j.get("res", j)

        # Common case: dict with "rec_texts"
        if isinstance(core, dict):
            rec_texts = core.get("rec_texts")
            if isinstance(rec_texts, list):
                lines.extend([t for t in rec_texts if isinstance(t, str)])
                continue

        # Sometimes: list of dicts (multi-page / multi-stage)
        if isinstance(core, list):
            for item in core:
                if isinstance(item, dict):
                    rec_texts = item.get("rec_texts")
                    if isinstance(rec_texts, list):
                        lines.extend([t for t in rec_texts if isinstance(t, str)])

    return "\n".join(lines)


def ocr_image(path: str) -> str:
    """OCR for image files using PaddleOCR 3.x."""
    print("Extracting from OCR image directly (PaddleOCR predict)")
    output = ocr.predict(input=path)  # NOTE: no cls=
    return paddle_predict_to_text(output)


def ocr_pdf_pages(path: str, page_numbers) -> dict[int, str]:
    """Render and OCR selected zero-based PDF page numbers."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for OCR fallback on PDFs. Install it with: pip install pymupdf"
        ) from e

    requested_pages = set(page_numbers)
    page_texts: dict[int, str] = {}

    with fitz.open(path) as pdf:
        invalid_pages = requested_pages.difference(range(len(pdf)))
        if invalid_pages:
            raise ValueError(f"PDF page numbers out of range: {sorted(invalid_pages)}")

        with tempfile.TemporaryDirectory() as temp_dir:
            for page_num in sorted(requested_pages):
                page = pdf[page_num]
                pix = page.get_pixmap()

                temp_img = os.path.join(temp_dir, f"page_{page_num}.png")
                pix.save(temp_img)

                output = ocr.predict(input=temp_img)
                page_texts[page_num] = paddle_predict_to_text(output).strip()

    return page_texts


def ocr_pdf(path: str) -> str:
    """Render and OCR every page in a PDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for OCR fallback on PDFs. Install it with: pip install pymupdf"
        ) from e

    with fitz.open(path) as pdf:
        page_numbers = list(range(len(pdf)))

    page_texts = ocr_pdf_pages(path, page_numbers)
  # Technical debt: PDF pages are currently rendered at PyMuPDF's default
  # resolution (about 72 DPI) before OCR. Increasing the render scale with
  # fitz.Matrix(2, 2) or a configurable DPI may improve recognition of small,
  # faint, or low-quality text, but it also increases processing time and memory
  # usage. Current OCR accuracy is acceptable, so treat this as a future
  # accuracy/performance option rather than a bug.
    return "\n\n".join(page_texts[page_num] for page_num in page_numbers)


def extract_docx(path: str) -> str:
    """Extract text from Word documents."""
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])

