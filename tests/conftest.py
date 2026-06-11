import sys
import types
from unittest.mock import MagicMock


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that use real models and sample files",
    )


def pytest_configure(config):
    # Stub the paddleocr module before any test imports agent.text_extraction,
    # so unit tests never load the real model. Integration runs skip the stub
    # and use the real package.
    if not config.getoption("--run-integration") and "paddleocr" not in sys.modules:
        fake = types.ModuleType("paddleocr")
        fake.PaddleOCR = MagicMock(return_value=MagicMock())
        sys.modules["paddleocr"] = fake
