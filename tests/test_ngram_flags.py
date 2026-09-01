"""Tests for the n-gram table quantization flags (-ngb/-ngf).

exllamav3 >= 1.4.5 quantizes hashed n-gram embedding tables (PLE models,
e.g. Qwen3.8-Flash-Next) during conversion. ezexl3 exposes --ngram_bits
and --ngram_file on repo/quantize and forwards them through the
quant_args passthrough, with a friendly up-front support check.
"""

import argparse
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ezexl3 import cli
from ezexl3 import quantize

REPO_ROOT = Path(__file__).resolve().parent.parent


class NgramCliWiringTests(unittest.TestCase):
    def test_repo_passes_ngram_bits_into_quant_args(self):
        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run:
            rc = cli.main(["repo", "-m", "/tmp/model", "-b", "4", "-ngb", "3", "-np"])
        self.assertEqual(rc, 0)
        self.assertEqual(mock_run.call_args.kwargs["quant_args"], ["-ngb", "3"])

    def test_ngram_bits_out_of_range_rejected(self):
        for bad in ("0", "9"):
            with self.assertRaises(SystemExit):
                cli.main(["repo", "-m", "/tmp/model", "-b", "4", "-ngb", bad, "-np"])

    def test_ngram_file_must_exist(self):
        with self.assertRaises(SystemExit):
            cli.main([
                "repo", "-m", "/tmp/model", "-b", "4",
                "-ngf", "/nonexistent/ngram.safetensors", "-np",
            ])

    def test_ngram_file_forwarded_as_absolute_path(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            with patch("ezexl3.repo.run_repo", return_value=0) as mock_run:
                cli.main(["repo", "-m", "/tmp/model", "-b", "4", "-ngf", f.name, "-np"])
            quant_args = mock_run.call_args.kwargs["quant_args"]
            self.assertEqual(quant_args[0], "-ngf")
            self.assertEqual(quant_args[1], os.path.abspath(f.name))

    def test_quantize_accepts_ngram_flags(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            ["quantize", "-m", "/tmp/model", "-b", "4", "-ngb", "4"]
        )
        self.assertEqual(args.ngram_bits, 4)


class NgramSupportCheckTests(unittest.TestCase):
    def test_run_one_refuses_ngb_on_old_exllamav3(self):
        # A convert parser without --ngram_bits stands in for exllamav3 < 1.4.5
        old_parser = argparse.ArgumentParser()
        old_parser.add_argument("-i")
        with tempfile.TemporaryDirectory() as model_dir:
            with patch("ezexl3.quantize._ensure_exl3_cal_data"), \
                 patch("ezexl3.quantize._get_exl3_convert",
                       return_value=(old_parser, None, None)):
                ok = quantize.run_one(
                    model_dir, "4", ["-ngb", "4"],
                    out_tmpl=os.path.join(model_dir, "{bpw}"),
                    w_tmpl=os.path.join(model_dir, "w-{bpw}"),
                    dry_run=False,
                )
        self.assertFalse(ok)


class NgramUiWiringTests(unittest.TestCase):
    def test_commands_js_has_ngram_group(self):
        src = (REPO_ROOT / "ezexl3" / "ui" / "static" / "js" / "commands.js").read_text()
        self.assertIn('flag: "-ngb"', src)
        self.assertIn('flag: "-ngf"', src)
        self.assertIn('group: "ngram"', src)

    def test_forms_js_renders_collapsed_groups(self):
        src = (REPO_ROOT / "ezexl3" / "ui" / "static" / "js" / "forms.js").read_text()
        self.assertIn("createGroupEl", src)
        self.assertIn("subsection-header", src)


if __name__ == "__main__":
    unittest.main()
