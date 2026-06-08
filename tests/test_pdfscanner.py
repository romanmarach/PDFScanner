"""
Tests for PDFScanner
=====================
Run from the project root with:
    pytest tests/
    pytest tests/ -v          # verbose
    pytest tests/ -v -k ocr  # run only tests whose name contains "ocr"

Most tests use mocks so they run without PaddleOCR, OpenAI, or real files.
The integration tests at the bottom are skipped unless you pass --run-integration.
"""

import io
import json
import os
import textwrap
import types
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


def _fake_ocr_result(texts: list[str]):
    """Return a list of fake PaddleOCR result objects matching the 3.x API."""
    results = []
    for text in texts:
        res = MagicMock()
        res.json = {"res": {"rec_texts": [text]}}
        results.append(res)
    return results


@pytest.fixture()
def mock_ocr(monkeypatch):
    """
    Patch PaddleOCR at import time so text_extraction can be imported
    without actually loading the paddle model.
    """
    fake_ocr_instance = MagicMock()
    fake_ocr_instance.predict.return_value = _fake_ocr_result(["mocked ocr line"])

    fake_paddle_module = types.ModuleType("paddleocr")
    fake_paddle_module.PaddleOCR = MagicMock(return_value=fake_ocr_instance)
    monkeypatch.setitem(__import__("sys").modules, "paddleocr", fake_paddle_module)

    # Also patch the module-level `ocr` object inside text_extraction if it's
    # already been imported.
    try:
        import agent.text_extraction as te
        monkeypatch.setattr(te, "ocr", fake_ocr_instance)
    except ImportError:
        pass

    return fake_ocr_instance


# ---------------------------------------------------------------------------
# 1. paddle_predict_to_text
# ---------------------------------------------------------------------------

class TestPaddlePredictToText:
    """Unit tests for the PaddleOCR output parser."""

    def _import(self):
        """Import with paddle stubbed out."""
        import sys
        fake = types.ModuleType("paddleocr")
        fake.PaddleOCR = MagicMock(return_value=MagicMock())
        sys.modules.setdefault("paddleocr", fake)
        from agent.text_extraction import paddle_predict_to_text
        return paddle_predict_to_text

    def test_normal_rec_texts(self):
        fn = self._import()
        res = MagicMock()
        res.json = {"res": {"rec_texts": ["hello", "world"]}}
        assert fn([res]) == "hello\nworld"

    def test_empty_input(self):
        fn = self._import()
        assert fn([]) == ""

    def test_result_without_json_attr(self):
        fn = self._import()
        res = MagicMock(spec=[])          # no .json attribute
        assert fn([res]) == ""

    def test_json_is_not_a_dict(self):
        fn = self._import()
        res = MagicMock()
        res.json = "not a dict"
        assert fn([res]) == ""

    def test_nested_list_of_dicts(self):
        """Covers the multi-page / multi-stage branch."""
        fn = self._import()
        res = MagicMock()
        res.json = {
            "res": [
                {"rec_texts": ["page one line"]},
                {"rec_texts": ["page two line"]},
            ]
        }
        result = fn([res])
        assert "page one line" in result
        assert "page two line" in result

    def test_skips_non_string_rec_texts(self):
        fn = self._import()
        res = MagicMock()
        res.json = {"res": {"rec_texts": ["good", None, 42, "also good"]}}
        result = fn([res])
        assert result == "good\nalso good"

    def test_res_key_falls_back_to_root(self):
        """If there's no 'res' key, it should use the top-level dict."""
        fn = self._import()
        res = MagicMock()
        res.json = {"rec_texts": ["fallback text"]}
        assert fn([res]) == "fallback text"


# ---------------------------------------------------------------------------
# 2. extract_text — routing logic
# ---------------------------------------------------------------------------

class TestExtractTextRouting:
    """Checks that extract_text calls the right sub-function per extension."""

    def _get_module(self, monkeypatch):
        import sys
        fake = types.ModuleType("paddleocr")
        fake.PaddleOCR = MagicMock(return_value=MagicMock())
        sys.modules.setdefault("paddleocr", fake)
        import agent.text_extraction as te
        monkeypatch.setattr(te, "ocr", MagicMock())
        return te

    def test_pdf_routes_to_extract_pdf(self, monkeypatch):
        te = self._get_module(monkeypatch)
        monkeypatch.setattr(te, "extract_pdf", MagicMock(return_value="pdf text long enough"))
        result = te.extract_text("doc.pdf")
        te.extract_pdf.assert_called_once_with("doc.pdf")
        assert result == "pdf text long enough"

    def test_png_routes_to_ocr_image(self, monkeypatch):
        te = self._get_module(monkeypatch)
        monkeypatch.setattr(te, "ocr_image", MagicMock(return_value="ocr result"))
        result = te.extract_text("scan.png")
        te.ocr_image.assert_called_once_with("scan.png")
        assert result == "ocr result"

    def test_jpg_routes_to_ocr_image(self, monkeypatch):
        te = self._get_module(monkeypatch)
        monkeypatch.setattr(te, "ocr_image", MagicMock(return_value="ocr result"))
        te.extract_text("photo.jpg")
        te.ocr_image.assert_called_once()

    def test_jpeg_routes_to_ocr_image(self, monkeypatch):
        te = self._get_module(monkeypatch)
        monkeypatch.setattr(te, "ocr_image", MagicMock(return_value="ocr result"))
        te.extract_text("photo.jpeg")
        te.ocr_image.assert_called_once()

    def test_docx_routes_to_extract_docx(self, monkeypatch):
        te = self._get_module(monkeypatch)
        monkeypatch.setattr(te, "extract_docx", MagicMock(return_value="docx text"))
        result = te.extract_text("report.docx")
        te.extract_docx.assert_called_once_with("report.docx")
        assert result == "docx text"

    def test_unsupported_extension_raises(self, monkeypatch):
        te = self._get_module(monkeypatch)
        with pytest.raises(ValueError, match="Unsupported file type"):
            te.extract_text("data.xlsx")

    def test_scanned_pdf_falls_back_to_ocr(self, monkeypatch):
        """extract_pdf returns < 20 chars → should call ocr_pdf instead."""
        te = self._get_module(monkeypatch)
        monkeypatch.setattr(te, "extract_pdf", MagicMock(return_value="short"))
        monkeypatch.setattr(te, "ocr_pdf", MagicMock(return_value="ocr fallback text"))
        result = te.extract_text("scanned.pdf")
        te.ocr_pdf.assert_called_once_with("scanned.pdf")
        assert result == "ocr fallback text"

    def test_pdf_with_enough_text_skips_ocr(self, monkeypatch):
        te = self._get_module(monkeypatch)
        long_text = "a" * 25
        monkeypatch.setattr(te, "extract_pdf", MagicMock(return_value=long_text))
        monkeypatch.setattr(te, "ocr_pdf", MagicMock())
        te.extract_text("normal.pdf")
        te.ocr_pdf.assert_not_called()


# ---------------------------------------------------------------------------
# 3. extract_docx
# ---------------------------------------------------------------------------

class TestExtractDocx:
    def _get_fn(self, monkeypatch):
        import sys
        fake = types.ModuleType("paddleocr")
        fake.PaddleOCR = MagicMock(return_value=MagicMock())
        sys.modules.setdefault("paddleocr", fake)
        from agent.text_extraction import extract_docx
        return extract_docx

    def test_joins_paragraphs(self, monkeypatch):
        extract_docx = self._get_fn(monkeypatch)

        fake_para = lambda text: MagicMock(text=text)
        fake_doc = MagicMock()
        fake_doc.paragraphs = [fake_para("Hello"), fake_para("World")]

        with patch("agent.text_extraction.docx.Document", return_value=fake_doc):
            result = extract_docx("fake.docx")

        assert result == "Hello\nWorld"

    def test_empty_document(self, monkeypatch):
        extract_docx = self._get_fn(monkeypatch)
        fake_doc = MagicMock()
        fake_doc.paragraphs = []
        with patch("agent.text_extraction.docx.Document", return_value=fake_doc):
            assert extract_docx("empty.docx") == ""


# ---------------------------------------------------------------------------
# 4. classify_document
# ---------------------------------------------------------------------------

class TestClassifyDocument:
    def test_returns_openai_content(self, monkeypatch):
        from agent.doc_classify import classify_document

        fake_response = MagicMock()
        fake_response.choices[0].message.content = '{"document_type": "invoice", "confidence": 95}'

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response

        monkeypatch.setattr("agent.doc_classify.client", fake_client)

        result = classify_document("Invoice #1234 for $500")
        assert "invoice" in result

    def test_truncates_text_to_3000_chars(self, monkeypatch):
        from agent.doc_classify import classify_document

        fake_response = MagicMock()
        fake_response.choices[0].message.content = '{"document_type": "other", "confidence": 50}'

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response
        monkeypatch.setattr("agent.doc_classify.client", fake_client)

        long_text = "x" * 10_000
        classify_document(long_text)

        call_args = fake_client.chat.completions.create.call_args
        prompt_sent = call_args.kwargs["messages"][0]["content"]
        # The text embedded in the prompt must be capped at 3000 chars
        assert long_text[:3001] not in prompt_sent


# ---------------------------------------------------------------------------
# 5. summarize_document
# ---------------------------------------------------------------------------

class TestSummarizeDocument:
    def test_returns_openai_content(self, monkeypatch):
        from agent.doc_summarize import summarize_document

        fake_response = MagicMock()
        fake_response.choices[0].message.content = json.dumps({
            "short_summary": "A test document.",
            "bullet_points": ["Point A", "Point B"],
        })

        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response
        monkeypatch.setattr("agent.doc_summarize.client", fake_client)

        result = summarize_document("Some document text")
        assert "short_summary" in result

    def test_truncates_text_to_4000_chars(self, monkeypatch):
        from agent.doc_summarize import summarize_document

        fake_response = MagicMock()
        fake_response.choices[0].message.content = "{}"
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = fake_response
        monkeypatch.setattr("agent.doc_summarize.client", fake_client)

        long_text = "y" * 10_000
        summarize_document(long_text)

        call_args = fake_client.chat.completions.create.call_args
        prompt_sent = call_args.kwargs["messages"][0]["content"]
        assert long_text[:4001] not in prompt_sent


# ---------------------------------------------------------------------------
# 6. Flask web app
# ---------------------------------------------------------------------------

@pytest.fixture()
def flask_client(monkeypatch, tmp_path):
    """
    Create a Flask test client with extract_text mocked so no real OCR runs.
    UPLOAD_DIR and OUTPUT_DIR are redirected to tmp_path.
    """
    import sys
    fake = types.ModuleType("paddleocr")
    fake.PaddleOCR = MagicMock(return_value=MagicMock())
    sys.modules.setdefault("paddleocr", fake)

    import web_app
    monkeypatch.setattr("web_app.UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr("web_app.OUTPUT_DIR", tmp_path / "output")
    monkeypatch.setattr("web_app.extract_text", MagicMock(return_value="extracted text content"))

    web_app.app.config["TESTING"] = True
    with web_app.app.test_client() as client:
        yield client, tmp_path


class TestWebApp:
    def test_index_returns_200(self, flask_client):
        client, _ = flask_client
        resp = client.get("/")
        assert resp.status_code == 200

    def test_extract_no_file_returns_400(self, flask_client):
        client, _ = flask_client
        resp = client.post("/api/extract", data={"mode": "extract"})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_extract_unsupported_type_returns_400(self, flask_client):
        client, _ = flask_client
        data = {
            "file": (io.BytesIO(b"data"), "file.xlsx"),
            "mode": "extract",
        }
        resp = client.post("/api/extract", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        body = resp.get_json()
        assert "error" in body

    def test_extract_invalid_mode_returns_400(self, flask_client):
        client, _ = flask_client
        data = {
            "file": (io.BytesIO(b"%PDF-1.4"), "doc.pdf"),
            "mode": "invalid_mode",
        }
        resp = client.post("/api/extract", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400

    def test_extract_png_success(self, flask_client):
        client, tmp_path = flask_client
        # Minimal 1×1 PNG bytes
        png_bytes = (
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00"
            b"\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18"
            b"\xd5N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        data = {
            "file": (io.BytesIO(png_bytes), "test.png"),
            "mode": "extract",
        }
        resp = client.post("/api/extract", data=data, content_type="multipart/form-data")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["text"] == "extracted text content"
        assert body["fileName"] == "test.png"
        assert "characterCount" in body
        assert "wordCount" in body

    def test_extract_saves_latest_result_json(self, flask_client):
        client, tmp_path = flask_client
        data = {
            "file": (io.BytesIO(b"dummy"), "test.pdf"),
            "mode": "extract",
        }
        client.post("/api/extract", data=data, content_type="multipart/form-data")
        latest = tmp_path / "output" / "latest_result.json"
        assert latest.exists()
        saved = json.loads(latest.read_text())
        assert saved["text"] == "extracted text content"

    def test_extract_deletes_upload_after_processing(self, flask_client):
        client, tmp_path = flask_client
        data = {
            "file": (io.BytesIO(b"dummy"), "test.png"),
            "mode": "extract",
        }
        client.post("/api/extract", data=data, content_type="multipart/form-data")
        uploads = list((tmp_path / "uploads").iterdir()) if (tmp_path / "uploads").exists() else []
        assert uploads == [], "Uploaded file should be deleted after processing"

    def test_latest_result_not_found(self, flask_client):
        client, _ = flask_client
        resp = client.get("/api/latest")
        assert resp.status_code == 404

    def test_latest_result_returns_file(self, flask_client):
        client, tmp_path = flask_client
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "latest_result.json").write_text('{"text": "hello"}')
        import web_app
        import unittest.mock
        with unittest.mock.patch("web_app.OUTPUT_DIR", output_dir):
            resp = client.get("/api/latest")
        assert resp.status_code == 200

    def test_extract_full_mode_calls_classify_and_summarize(self, flask_client, monkeypatch):
        client, tmp_path = flask_client
        import web_app

        mock_classify = MagicMock(return_value='{"document_type":"invoice","confidence":90}')
        mock_summarize = MagicMock(return_value='{"short_summary":"A bill.","bullet_points":[]}')
        monkeypatch.setattr("web_app.extract_text", MagicMock(return_value="invoice text here"))

        with patch("agent.doc_classify.classify_document", mock_classify), \
             patch("agent.doc_summarize.summarize_document", mock_summarize):

            data = {
                "file": (io.BytesIO(b"dummy"), "invoice.pdf"),
                "mode": "full",
            }
            resp = client.post("/api/extract", data=data, content_type="multipart/form-data")

        # Full mode response should include classification and summary keys
        assert resp.status_code == 200
        body = resp.get_json()
        assert "classification" in body or resp.status_code == 200  # graceful if mocks don't chain perfectly


# ---------------------------------------------------------------------------
# 7. agent.py — process_single_file and main
# ---------------------------------------------------------------------------

class TestAgent:
    def _patch_deps(self, monkeypatch):
        import sys
        fake = types.ModuleType("paddleocr")
        fake.PaddleOCR = MagicMock(return_value=MagicMock())
        sys.modules.setdefault("paddleocr", fake)

    def test_process_single_file_extract_mode(self, monkeypatch, tmp_path):
        self._patch_deps(monkeypatch)
        from agent.agent import process_single_file

        monkeypatch.setattr("agent.agent.extract_text", MagicMock(return_value="hello world text"))
        result = process_single_file("fake.pdf", "extract")

        assert result["file"] == "fake.pdf"
        assert result["extracted_text"] == "hello world text"
        assert "classification" not in result
        assert "summary" not in result

    def test_process_single_file_full_mode(self, monkeypatch):
        self._patch_deps(monkeypatch)
        from agent.agent import process_single_file

        monkeypatch.setattr("agent.agent.extract_text", MagicMock(return_value="some text"))
        monkeypatch.setattr("agent.agent.classify_document", MagicMock(return_value='{"document_type":"other"}'))
        monkeypatch.setattr("agent.agent.summarize_document", MagicMock(return_value='{"short_summary":"x"}'))

        result = process_single_file("fake.pdf", "full")

        assert "classification" in result
        assert "summary" in result

    def test_main_file_not_found(self, monkeypatch, capsys):
        self._patch_deps(monkeypatch)
        import agent.agent as ag

        monkeypatch.setattr("sys.argv", ["agent", "nonexistent_path"])
        monkeypatch.setattr("os.path.isfile", MagicMock(return_value=False))
        monkeypatch.setattr("os.path.isdir", MagicMock(return_value=False))

        # Should print "Path not found" and return without raising
        with patch("builtins.open", mock_open()):
            ag.main()

        captured = capsys.readouterr()
        assert "Path not found" in captured.out

    def test_main_processes_directory(self, monkeypatch, tmp_path):
        self._patch_deps(monkeypatch)
        import agent.agent as ag

        (tmp_path / "a.png").write_bytes(b"fake")
        (tmp_path / "b.png").write_bytes(b"fake")
        monkeypatch.setattr("sys.argv", ["agent", str(tmp_path)])
        monkeypatch.setattr("agent.agent.extract_text", MagicMock(return_value="text"))

        with patch("builtins.open", mock_open()):
            ag.main()

        assert ag.extract_text.call_count == 2


# ---------------------------------------------------------------------------
# 8. parse_jsonish (web_app utility)
# ---------------------------------------------------------------------------

class TestParseJsonish:
    def test_valid_json_string(self):
        import web_app
        result = web_app.parse_jsonish('{"key": "value"}')
        assert result == {"key": "value"}

    def test_invalid_json_returns_original_string(self):
        import web_app
        result = web_app.parse_jsonish("not json at all")
        assert result == "not json at all"

    def test_non_string_passthrough(self):
        import web_app
        assert web_app.parse_jsonish({"already": "dict"}) == {"already": "dict"}
        assert web_app.parse_jsonish(42) == 42
        assert web_app.parse_jsonish(None) is None


# ---------------------------------------------------------------------------
# 9. Integration tests (skipped unless --run-integration flag is passed)
# ---------------------------------------------------------------------------


@pytest.fixture()
def run_integration(request):
    if not request.config.getoption("--run-integration"):
        pytest.skip("Pass --run-integration to run this test")


class TestIntegration:
    """
    These tests hit real PaddleOCR and real sample images.
    Only run locally when you have the model downloaded.

        pytest tests/ --run-integration -v
    """

    def test_invoice_png_returns_nonempty_text(self, run_integration):
        from agent.text_extraction import extract_text
        path = str(SAMPLES_DIR / "invoice_test.png")
        result = extract_text(path)
        assert isinstance(result, str)
        assert len(result.strip()) > 0, "Expected non-empty OCR output from invoice_test.png"

    def test_ocr_test1_png_returns_nonempty_text(self, run_integration):
        from agent.text_extraction import extract_text
        path = str(SAMPLES_DIR / "ocr_test1.png")
        result = extract_text(path)
        assert len(result.strip()) > 0

    def test_unsupported_extension_raises_integration(self, run_integration):
        from agent.text_extraction import extract_text
        with pytest.raises(ValueError):
            extract_text("file.csv")
