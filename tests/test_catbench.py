import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ezexl3.catbench import (
    _safetensors_size_gib,
    extract_svg,
    _run_matplotlib_code,
)
from ezexl3.readme import _build_catbench_grid, run_readme
from ezexl3.repo import _catbench_file_prefix


class CatbenchFilePrefixTests(unittest.TestCase):
    def test_integer_bpw(self):
        self.assertEqual(_catbench_file_prefix("4"), "4.00bpw")

    def test_decimal_bpw(self):
        self.assertEqual(_catbench_file_prefix("3.5"), "3.50bpw")

    def test_bf16_label(self):
        self.assertEqual(_catbench_file_prefix("bf16"), "bf16")

    def test_base_label(self):
        self.assertEqual(_catbench_file_prefix("base"), "bf16")


class SafetensorsSizeTests(unittest.TestCase):
    def test_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertAlmostEqual(_safetensors_size_gib(tmpdir), 0.0)

    def test_nonexistent_dir(self):
        self.assertAlmostEqual(_safetensors_size_gib("/nonexistent/path"), 0.0)

    def test_with_safetensors_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake 1 MiB safetensors file
            path = os.path.join(tmpdir, "model.safetensors")
            with open(path, "wb") as f:
                f.write(b"\x00" * (1024 * 1024))  # 1 MiB
            size = _safetensors_size_gib(tmpdir)
            self.assertAlmostEqual(size, 1.0 / 1024, places=6)


class SVGExtractionTests(unittest.TestCase):
    def test_extract_raw_svg(self):
        text = 'Some text\n<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>\nmore text'
        result = extract_svg(text)
        self.assertIsNotNone(result)
        self.assertIn("<svg", result)
        self.assertIn("</svg>", result)

    def test_no_svg_returns_none(self):
        text = "Just some plain text without any SVG content."
        result = extract_svg(text)
        self.assertIsNone(result)

    def test_extract_from_code_block_without_matplotlib(self):
        text = "```python\nprint('hello')\n```"
        result = extract_svg(text)
        self.assertIsNone(result)


class CatbenchGridTests(unittest.TestCase):
    def test_empty_catbench_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _build_catbench_grid(tmpdir)
            self.assertEqual(result, "")

    def test_no_catbench_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _build_catbench_grid(tmpdir)
            self.assertEqual(result, "")

    def test_grid_with_svgs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catdir = os.path.join(tmpdir, "catbench")
            os.makedirs(catdir)

            # Create canonical SVGs
            for name in ["2.00bpw.svg", "3.00bpw.svg", "4.00bpw.svg", "bf16.svg"]:
                Path(os.path.join(catdir, name)).write_text('<svg xmlns="test"></svg>')

            # Create a numbered variant that should be excluded
            Path(os.path.join(catdir, "2.00bpw_1.svg")).write_text('<svg xmlns="test"></svg>')

            result = _build_catbench_grid(tmpdir)
            self.assertIn('<table align="center">', result)
            self.assertIn("2.00 bpw", result)
            self.assertIn("3.00 bpw", result)
            self.assertIn("4.00 bpw", result)
            self.assertIn("BF16", result)
            self.assertIn('catbench/2.00bpw.svg', result)
            self.assertIn('width="160"', result)
            # Numbered variant should not appear
            self.assertNotIn("2.00bpw_1.svg", result)

    def test_grid_sorting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catdir = os.path.join(tmpdir, "catbench")
            os.makedirs(catdir)

            for name in ["6.00bpw.svg", "2.00bpw.svg", "bf16.svg"]:
                Path(os.path.join(catdir, name)).write_text('<svg></svg>')

            result = _build_catbench_grid(tmpdir)
            # 2.00 should come before 6.00, bf16 last
            idx_2 = result.index("2.00 bpw")
            idx_6 = result.index("6.00 bpw")
            idx_bf = result.index("BF16")
            self.assertLess(idx_2, idx_6)
            self.assertLess(idx_6, idx_bf)

    def test_four_columns_per_row(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            catdir = os.path.join(tmpdir, "catbench")
            os.makedirs(catdir)

            # 5 SVGs should produce 2 rows (4 + 1)
            for i in range(1, 6):
                Path(os.path.join(catdir, f"{i}.00bpw.svg")).write_text('<svg></svg>')

            result = _build_catbench_grid(tmpdir)
            # Count <tr> tags
            tr_count = result.count("<tr>")
            self.assertEqual(tr_count, 2)


class CLICatbenchArgTests(unittest.TestCase):
    def test_catbench_flag_defaults(self):
        from ezexl3.cli import build_parser
        parser = build_parser()

        # No -cb flag: default 0
        args = parser.parse_args(["repo", "-m", "/tmp/model", "-b", "2"])
        self.assertEqual(args.catbench, 0)

    def test_catbench_flag_no_value(self):
        from ezexl3.cli import build_parser
        parser = build_parser()

        # -cb with no value: const=3
        args = parser.parse_args(["repo", "-m", "/tmp/model", "-b", "2", "-cb"])
        self.assertEqual(args.catbench, 3)

    def test_catbench_flag_with_value(self):
        from ezexl3.cli import build_parser
        parser = build_parser()

        # -cb 5: explicit value
        args = parser.parse_args(["repo", "-m", "/tmp/model", "-b", "2", "-cb", "5"])
        self.assertEqual(args.catbench, 5)

    def test_catbench_on_measure_subcommand(self):
        from ezexl3.cli import build_parser
        parser = build_parser()

        args = parser.parse_args(["measure", "-m", "/tmp/model", "-b", "2", "-cb"])
        self.assertEqual(args.catbench, 3)


class ReadmeCatbenchIntegrationTests(unittest.TestCase):
    def test_readme_includes_catbench_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "TestAuthor-TestModel"
            model_dir.mkdir()

            # Create measurement CSV
            csv_path = model_dir / "TestAuthor-TestModelMeasured.csv"
            csv_path.write_text(
                "weights,KL Div,PPL r-100,GiB\n"
                "4,0.1234,7.89,5.43\n"
                "bf16,0,6.54,10.0\n"
            )

            # Create catbench SVGs
            catdir = model_dir / "catbench"
            catdir.mkdir()
            (catdir / "4.00bpw.svg").write_text('<svg xmlns="test"><circle/></svg>')
            (catdir / "bf16.svg").write_text('<svg xmlns="test"><rect/></svg>')

            run_readme(str(model_dir), template_name="basic", interactive=False, include_catbench=True)

            readme = (model_dir / "README.md").read_text()
            self.assertIn("SVG Catbench", readme)
            self.assertIn("catbench/4.00bpw.svg", readme)
            self.assertIn("catbench/bf16.svg", readme)

    def test_readme_no_catbench_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "TestAuthor-TestModel"
            model_dir.mkdir()

            csv_path = model_dir / "TestAuthor-TestModelMeasured.csv"
            csv_path.write_text(
                "weights,KL Div,PPL r-100,GiB\n"
                "4,0.1234,7.89,5.43\n"
                "bf16,0,6.54,10.0\n"
            )

            run_readme(str(model_dir), template_name="basic", interactive=False, include_catbench=False)

            readme = (model_dir / "README.md").read_text()
            self.assertNotIn("SVG Catbench", readme)


if __name__ == "__main__":
    unittest.main()
