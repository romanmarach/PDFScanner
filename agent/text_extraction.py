import pdfplumber
import pytesseract
from PIL import Image
import docx
import os
import tempfile
import shutil


# Auto-detect Tesseract path from PATH, fallback to common locations
tesseract_path = shutil.which("tesseract")
if not tesseract_path:
    for path in [r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                 r"D:\Program Files\Tesseract-OCR\tesseract.exe"]:
        if os.path.exists(path):
            tesseract_path = path
            break

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    raise RuntimeError("Tesseract not found. Please install it and add to PATH.")
#==============================================================

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
    
#understand everything up until here
#______________________________________________________________________


def extract_pdf(path):
    """Extract text from a normal PDF (no OCR)."""
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

#______________________________________________________________________

def ocr_pdf(path: str) -> str:
    """Convert each PDF page to an image and run OCR."""
    try:
        import fitz  # PyMuPDF for PDF → image
    except ImportError as e:
        raise ImportError(
            "PyMuPDF is required for OCR on PDFs. Install it with: pip install pymupdf"
        ) from e
#______________________________________________________________________
#main ocr code. 
    text = ""
    with fitz.open(path) as pdf:
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            pix = page.get_pixmap()
            temp_img = os.path.join(tempfile.gettempdir(), f"page_{page_num}.png")
            pix.save(temp_img)
            text += pytesseract.image_to_string(Image.open(temp_img))

    return text
#turn image pdfs into files, save the file then run the extraction on it.
# PDF → page → render to image → save image → OCR image → text
#______________________________________________________________________


def ocr_image(path):
    """OCR for image files."""
    print("Extracting from ocr image directly")
    return pytesseract.image_to_string(Image.open(path))


def extract_docx(path):
    """Extract text from Word documents."""
    doc = docx.Document(path)
    return "\n".join([para.text for para in doc.paragraphs])
