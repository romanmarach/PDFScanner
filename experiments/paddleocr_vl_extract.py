"""Standalone PaddleOCR-VL text extraction experiment."""

import argparse
import os
import sys
from html.parser import HTMLParser
from pathlib import Path

# Avoid network source checks. The model still fails clearly if it is not cached.
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE_CHECK", "False")


class PlainTextParser(HTMLParser):
    """Convert PaddleOCR-VL's table HTML into readable plain text."""

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr" and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")
        elif tag == "td" and self.parts and not self.parts[-1].endswith(("\n", "\t")):
            self.parts.append("\t")

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return "".join(self.parts).strip()


def to_plain_text(markdown_text: str) -> str:
    parser = PlainTextParser()
    parser.feed(markdown_text)
    return parser.text()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract text from an image or PDF with PaddleOCR-VL."
    )
    parser.add_argument("input", type=Path, help="Path to an image or PDF")
    parser.add_argument(
        "--output",
        type=Path,
        help="Text output path (default: output/paddleocr_vl/<name>.txt)",
    )
    return parser.parse_args()


def extract_text(input_path: Path) -> str:
    try:
        from paddleocr import PaddleOCRVL
    except ImportError as exc:
        raise RuntimeError(
            "PaddleOCR-VL is unavailable. Install a PaddleOCR 3.x release that "
            "includes PaddleOCRVL."
        ) from exc

    pipeline = PaddleOCRVL(
        vl_rec_backend="native",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_chart_recognition=False,
    )
    results = pipeline.predict(input=str(input_path))

    pages = []
    for result in results:
        markdown = result.markdown
        page_text = markdown.get("markdown_texts", "").strip()
        if page_text:
            pages.append(to_plain_text(page_text))

    return "\n\n".join(pages)


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    output_path = args.output
    if output_path is None:
        output_path = (
            Path("output") / "paddleocr_vl" / f"{input_path.stem}.txt"
        )
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        text = extract_text(input_path)
    except Exception as exc:
        print(f"PaddleOCR-VL failed: {exc}", file=sys.stderr)
        return 1

    output_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nSaved extracted text to: {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
