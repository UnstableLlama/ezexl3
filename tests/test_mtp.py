"""Tests for the MTP tensor quant-only stage (ezexl3 mtp).

Covers CLI flag wiring into run_mtp, the vendored-script command
construction, skip-if-done behavior, and the WebUI wiring (server
allowlist, commands.js schema, nav button).
"""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from ezexl3 import cli
from ezexl3 import mtp

REPO_ROOT = Path(__file__).resolve().parent.parent


class MtpCliParserTests(unittest.TestCase):
    def test_defaults(self):
        parser = cli.build_parser()
        args = parser.parse_args(["mtp", "-m", "/tmp/model"])
        self.assertEqual(args.cmd, "mtp")
        self.assertEqual(args.mtp_bits, 4)
        self.assertIsNone(args.out_file)
        self.assertEqual(args.device, 0)

    def test_explicit_flags(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            ["mtp", "-m", "/tmp/model", "-mb", "6", "-o", "/tmp/out.safetensors", "-d", "1"]
        )
        self.assertEqual(args.mtp_bits, 6)
        self.assertEqual(args.out_file, "/tmp/out.safetensors")
        self.assertEqual(args.device, 1)


class MtpCliDispatchTests(unittest.TestCase):
    def test_cli_dispatch_calls_run_mtp(self):
        with patch("ezexl3.mtp.run_mtp", return_value=0) as mock_run:
            rc = cli.main(["mtp", "-m", "/tmp/model", "-mb", "3", "-d", "1"])
        self.assertEqual(rc, 0)
        mock_run.assert_called_once_with(
            model_dir="/tmp/model",
            mtp_bits=3,
            out_file=None,
            device=1,
            hq=False,
        )

    def test_cli_dispatch_passes_hq(self):
        with patch("ezexl3.mtp.run_mtp", return_value=0) as mock_run:
            rc = cli.main(["mtp", "-m", "/tmp/model", "-hq"])
        self.assertEqual(rc, 0)
        self.assertTrue(mock_run.call_args.kwargs["hq"])

    def test_out_file_rejected_with_multiple_models(self):
        with self.assertRaises(SystemExit):
            cli.main(["mtp", "-m", "/tmp/a", "/tmp/b", "-o", "/tmp/out.safetensors"])

    def test_run_mtp_error_returns_nonzero(self):
        with patch("ezexl3.mtp.run_mtp", side_effect=RuntimeError("boom")):
            rc = cli.main(["mtp", "-m", "/tmp/model"])
        self.assertEqual(rc, 1)


class MtpRunnerTests(unittest.TestCase):
    def test_builds_vendored_script_command(self):
        import tempfile
        with tempfile.TemporaryDirectory() as model_dir:
            with patch("ezexl3.mtp.run_cmd_capture", return_value="") as mock_cmd:
                rc = mtp.run_mtp(model_dir, mtp_bits=5, device=2)
            self.assertEqual(rc, 0)
            cmd = mock_cmd.call_args[0][0]
            self.assertTrue(cmd[1].endswith(os.path.join("vendor", "convert_mtp.py")))
            self.assertEqual(cmd[cmd.index("-m") + 1], os.path.abspath(model_dir))
            self.assertEqual(cmd[cmd.index("-mb") + 1], "5")
            self.assertEqual(cmd[cmd.index("-d") + 1], "2")
            out_file = cmd[cmd.index("-o") + 1]
            self.assertEqual(out_file, mtp.default_mtp_out_file(model_dir, 5))
            # Output dir is created ahead of the subprocess
            self.assertTrue(os.path.isdir(os.path.dirname(out_file)))

    def test_default_out_file_is_outside_model_root(self):
        out = mtp.default_mtp_out_file("/tmp/model", 4)
        self.assertEqual(
            out, os.path.join("/tmp/model", "mtp-quant", "mtp_4bpw.safetensors")
        )

    def test_default_out_file_hq_suffix(self):
        out = mtp.default_mtp_out_file("/tmp/model", 4, hq=True)
        self.assertEqual(
            out, os.path.join("/tmp/model", "mtp-quant", "mtp_4bpw_hq.safetensors")
        )

    def test_hq_flag_appended_to_vendored_command(self):
        import tempfile
        with tempfile.TemporaryDirectory() as model_dir:
            with patch("ezexl3.mtp.run_cmd_capture", return_value="") as mock_cmd:
                mtp.run_mtp(model_dir, mtp_bits=4, hq=True)
            cmd = mock_cmd.call_args[0][0]
            self.assertIn("-hq", cmd)

    def test_skips_when_output_exists(self):
        import tempfile
        with tempfile.TemporaryDirectory() as model_dir:
            out_file = mtp.default_mtp_out_file(model_dir, 4)
            os.makedirs(os.path.dirname(out_file))
            Path(out_file).write_bytes(b"")
            with patch("ezexl3.mtp.run_cmd_capture") as mock_cmd:
                rc = mtp.run_mtp(model_dir, mtp_bits=4)
            self.assertEqual(rc, 0)
            mock_cmd.assert_not_called()

    def test_missing_model_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            mtp.run_mtp("/nonexistent/model/dir")


class MtpVendorTests(unittest.TestCase):
    def test_vendored_script_present_and_in_manifest(self):
        vendor_dir = REPO_ROOT / "ezexl3" / "vendor"
        self.assertTrue((vendor_dir / "convert_mtp.py").is_file())
        manifest = json.loads((vendor_dir / "VENDOR_MANIFEST.json").read_text())
        self.assertIn("convert_mtp.py", manifest)


class MtpUiWiringTests(unittest.TestCase):
    """MTP has no tab of its own: the bitrate is an option on Quantize and
    Repo, since exllamav3's convert takes -mb directly. The standalone
    `ezexl3 mtp` command survives for retrofitting quants built before MTP
    support existed, but it is not reachable from the dashboard."""

    def test_server_does_not_allowlist_mtp(self):
        src = (REPO_ROOT / "ezexl3" / "ui" / "server.py").read_text()
        self.assertNotRegex(src, r'valid_commands = \{[^}]*"mtp"')

    def test_commands_js_has_no_mtp_tab(self):
        src = (REPO_ROOT / "ezexl3" / "ui" / "static" / "js" / "commands.js").read_text()
        self.assertNotIn("mtp: {", src)

    def test_mtp_bits_is_an_option_on_quantize_and_repo(self):
        src = (REPO_ROOT / "ezexl3" / "ui" / "static" / "js" / "commands.js").read_text()
        self.assertEqual(src.count('name: "mtp_bits", flag: "-mb"'), 2)
        self.assertEqual(src.count('label: "MTP Bits"'), 2)

    def test_index_html_has_no_mtp_nav_button(self):
        src = (REPO_ROOT / "ezexl3" / "ui" / "static" / "index.html").read_text()
        self.assertNotIn('data-cmd="mtp"', src)


if __name__ == "__main__":
    unittest.main()
