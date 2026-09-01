"""Tests for -ngr / ngram_ram: loading a PLE model's hashed n-gram
embedding table into system RAM instead of streaming it from disk.

Covers the chat CLI passthrough, the -ngr gating in _build_model_args
(only emitted when the installed exllamav3 exposes the flag), the
support probe, and the chat UI wiring.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock heavy dependencies BEFORE any project imports (same approach as
# test_chat_ngram.py) so no GPU install is required.
if "torch" not in sys.modules:
    _mock_torch = MagicMock()
    _mock_torch.cuda.is_available.return_value = False
    _mock_torch.cuda.device_count.return_value = 0
    sys.modules["torch"] = _mock_torch

if "exllamav3" not in sys.modules:
    sys.modules["exllamav3"] = MagicMock()

from ezexl3 import cli  # noqa: E402
from ezexl3.chat import inference  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeModelInitOld:
    """Parser without -ngr (exllamav3 < 1.4.5)."""

    @staticmethod
    def add_args(parser, cache=False):
        parser.add_argument("-m", "--model_dir")
        parser.add_argument("-gs", "--gpu_split")
        parser.add_argument("-cs", "--cache_size", type=int)
        parser.add_argument("-cq", "--cache_quant")


class FakeModelInitNew(FakeModelInitOld):
    """Parser with -ngr (exllamav3 >= 1.4.5)."""

    @staticmethod
    def add_args(parser, cache=False):
        FakeModelInitOld.add_args(parser, cache)
        parser.add_argument("-ngr", "--ngram_ram", action="store_true")


class NgramRamCliTests(unittest.TestCase):
    def test_chat_cli_passes_ngram_ram(self):
        with patch("ezexl3.chat.server.run_server") as mock_run:
            rc = cli.main(["chat", "-m", "/tmp/model", "-ngr", "--no-browser"])
        self.assertEqual(rc, 0)
        self.assertTrue(mock_run.call_args.kwargs["ngram_ram"])

    def test_chat_cli_defaults_ngram_ram_off(self):
        with patch("ezexl3.chat.server.run_server") as mock_run:
            cli.main(["chat", "--no-browser"])
        self.assertFalse(mock_run.call_args.kwargs["ngram_ram"])


class BuildModelArgsNgramRamTests(unittest.TestCase):
    def test_ngr_emitted_when_supported(self):
        with patch.object(inference, "_import_model_init", return_value=FakeModelInitNew):
            args = inference._build_model_args(
                "/m", [0], None, 32768, "6,6", ngram_ram=True)
        self.assertTrue(args.ngram_ram)

    def test_ngr_off_by_default(self):
        with patch.object(inference, "_import_model_init", return_value=FakeModelInitNew):
            args = inference._build_model_args("/m", [0], None, 32768, "6,6")
        self.assertFalse(args.ngram_ram)

    def test_ngr_dropped_on_old_build(self):
        # Requesting ngram_ram on an old build must not crash parse_args
        with patch.object(inference, "_import_model_init", return_value=FakeModelInitOld):
            args = inference._build_model_args(
                "/m", [0], None, 32768, "6,6", ngram_ram=True)
        self.assertFalse(getattr(args, "ngram_ram", False))


class NgramRamSupportTests(unittest.TestCase):
    def test_supported(self):
        with patch.object(inference, "_import_model_init", return_value=FakeModelInitNew):
            self.assertTrue(inference.ngram_ram_support())

    def test_unsupported(self):
        with patch.object(inference, "_import_model_init", return_value=FakeModelInitOld):
            self.assertFalse(inference.ngram_ram_support())

    def test_broken_import(self):
        with patch.object(inference, "_import_model_init", side_effect=ImportError):
            self.assertFalse(inference.ngram_ram_support())


class NgramRamEngineTests(unittest.TestCase):
    def test_engine_stores_flag(self):
        engine = inference.ChatEngine(ngram_ram=True)
        self.assertTrue(engine.ngram_ram)
        self.assertFalse(inference.ChatEngine().ngram_ram)


class NgramRamUiWiringTests(unittest.TestCase):
    def test_chat_index_has_checkbox(self):
        src = (REPO_ROOT / "ezexl3" / "chat" / "static" / "index.html").read_text()
        self.assertIn('id="ngram-ram-checkbox"', src)
        self.assertIn('id="ngram-ram-unsupported"', src)

    def test_model_js_sends_flag_and_gates_support(self):
        src = (REPO_ROOT / "ezexl3" / "chat" / "static" / "js" / "model.js").read_text()
        self.assertIn("ngram_ram:", src)
        self.assertIn("data.ngram_ram", src)


if __name__ == "__main__":
    unittest.main()
