import sys
import unittest
from unittest.mock import MagicMock, patch

# chat.server (imported by the chat dispatch) needs these mocked to stay
# GPU-free; other chat tests install the same mocks.
if "torch" not in sys.modules:
    _mock_torch = MagicMock()
    _mock_torch.cuda.is_available.return_value = False
    _mock_torch.cuda.device_count.return_value = 0
    sys.modules["torch"] = _mock_torch
if "exllamav3" not in sys.modules:
    sys.modules["exllamav3"] = MagicMock()

from ezexl3 import cli


class CliValidationTests(unittest.TestCase):
    def test_parse_devices_rejects_non_integer(self):
        with self.assertRaises(SystemExit):
            cli._parse_devices(["0", "gpu1"])

    def test_parse_devices_rejects_empty(self):
        with self.assertRaises(SystemExit):
            cli._parse_devices([])

    def test_parse_device_ratios_rejects_length_mismatch(self):
        with self.assertRaises(SystemExit):
            cli._parse_device_ratios(["1"], [0, 1])

    def test_parse_device_ratios_rejects_non_positive(self):
        with self.assertRaises(SystemExit):
            cli._parse_device_ratios(["1", "0"], [0, 1])

    def test_parse_device_ratios_accepts_valid(self):
        out = cli._parse_device_ratios(["1", "1.5"], [0, 1])
        self.assertEqual(out, ["1", "1.5"])

    def test_parse_layers_accepts_valid(self):
        self.assertEqual(cli._parse_layers(1), 1)
        self.assertEqual(cli._parse_layers(2), 2)
        self.assertEqual(cli._parse_layers(3), 3)

    def test_parse_layers_rejects_invalid(self):
        with self.assertRaises(SystemExit):
            cli._parse_layers(4)


class ChatDraftCliTests(unittest.TestCase):
    """`ezexl3 chat` can preload a draft source together with the model
    (required for recurrent models like Qwen3.5/3.6)."""

    def _run(self, argv):
        calls = {}

        def fake_run_server(**kwargs):
            calls.update(kwargs)

        with patch("ezexl3.chat.server.run_server", fake_run_server):
            rc = cli.main(argv)
        return rc, calls

    def test_draft_model_passed_through(self):
        rc, calls = self._run(
            ["chat", "-m", "/models/qwen", "-df", "/models/dflash",
             "--no-browser"])
        self.assertEqual(calls["model_dir"], "/models/qwen")
        self.assertEqual(calls["draft_model_dir"], "/models/dflash")
        self.assertFalse(calls["use_mtp"])
        self.assertEqual(calls["ngram_min"], 0)

    def test_mtp_and_ngram_passed_through(self):
        _, calls = self._run(["chat", "-m", "/models/qwen", "--mtp"])
        self.assertTrue(calls["use_mtp"])
        _, calls = self._run(["chat", "-m", "/models/qwen", "--ngram", "3"])
        self.assertEqual(calls["ngram_min"], 3)

    def test_multiple_draft_sources_rejected(self):
        rc, calls = self._run(
            ["chat", "-m", "/models/qwen", "-df", "/d", "--mtp"])
        self.assertEqual(rc, 2)
        self.assertEqual(calls, {})

    def test_draft_without_model_rejected(self):
        for extra in (["-df", "/d"], ["--mtp"], ["--ngram", "3"]):
            rc, calls = self._run(["chat"] + extra)
            self.assertEqual(rc, 2, extra)
            self.assertEqual(calls, {}, extra)


if __name__ == "__main__":
    unittest.main()
