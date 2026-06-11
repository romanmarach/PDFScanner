import os
import tempfile
import pdfplumber
import docx
from paddleocr import PaddleOCR
os.environ["PADDLE_PDX_MODEL_SOURCE_CHECK"] = "True"


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
        text = extract_pdf(file_path)

        # If PDF text is empty → fallback to OCR
        if len(text.strip()) < 20:
            print("⚠️ PDF appears scanned — using OCR fallback...")
            text = ocr_pdf(file_path)

        return text

    elif ext in [".jpg", ".jpeg", ".png"]:
        return ocr_image(file_path)

    elif ext == ".docx":
        return extract_docx(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def extract_pdf(path: str) -> str:
    """Extract text from a normal PDF (no OCR)."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


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


def ocr_pdf(path: str) -> str:
    """
    OCR for PDFs using PaddleOCR 3.x.
    1) Try passing the PDF path directly to ocr.predict()
    2) If that fails / returns empty, render pages with PyMuPDF and OCR images
    """
    print("Extracting from OCR PDF (PaddleOCR predict)")

    # Attempt 1: direct PDF OCR
    try:
        output = ocr.predict(input=path)
        text = paddle_predict_to_text(output)
        if text.strip():
            return text
        print("⚠️ Direct PDF OCR returned empty — falling back to page rendering...")
    except Exception as e:
        print(f"⚠️ Direct PDF OCR failed ({type(e).__name__}: {e}) — falling back to page rendering...")

    # Attempt 2: render pages to images (fallback)
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for OCR fallback on PDFs. Install it with: pip install pymupdf"
        ) from e

    combined_pages: list[str] = []
    with fitz.open(path) as pdf:
        with tempfile.TemporaryDirectory() as temp_dir:
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                pix = page.get_pixmap()

                temp_img = os.path.join(temp_dir, f"page_{page_num}.png")
                pix.save(temp_img)

                output = ocr.predict(input=temp_img)
                page_text = paddle_predict_to_text(output).strip()
                if page_text:
                    combined_pages.append(page_text)

    return "\n\n".join(combined_pages)


def extract_docx(path: str) -> str:
    """Extract text from Word documents."""
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])
