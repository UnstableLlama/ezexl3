import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from ezexl3.catbench import (
    _safetensors_size_gib,
    extract_svg,
    _run_matplotlib_code,
    build_prompt_and_stops,
    CATBENCH_PROMPT,
)
from ezexl3.chat.templates import infer_mode, infer_mode_from_path
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

            # Catbench panel must appear before CLI Download panel
            catbench_pos = readme.find("SVG Catbench")
            cli_pos = readme.find("CLI Download")
            self.assertGreater(cli_pos, catbench_pos,
                               "SVG Catbench panel should appear before CLI Download")

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


class InferModeTests(unittest.TestCase):
    def test_qwen35(self):
        self.assertEqual(infer_mode("Qwen3.5-32B-Instruct"), "qwen35")
        self.assertEqual(infer_mode("qwen3-5-7b"), "qwen35")

    def test_qwen36_maps_to_qwen35(self):
        # qwen3.6 uses the same reasoning-aware ChatML as qwen3.5
        self.assertEqual(infer_mode("Qwen3.6-27B"), "qwen35")

    def test_qwen36_dash_not_conflated_with_qwen35(self):
        # "qwen3-6" is a different family; must not map to qwen35
        self.assertNotEqual(infer_mode("qwen3-6-some-model"), "qwen35")

    def test_gemma4_before_gemma(self):
        self.assertEqual(infer_mode("gemma4-9b"), "gemma4")
        self.assertEqual(infer_mode("gemma-2-9b"), "gemma")

    def test_unknown_falls_back_to_chatml(self):
        self.assertEqual(infer_mode("some-random-model-name"), "chatml")

    def test_infer_from_path_bpw_subfolder(self):
        # A BPW subfolder basename doesn't match hints, so we fall back
        # to the parent directory's name.
        path = "/models/Qwen3.6-27B-exl3/2.50bpw"
        self.assertEqual(infer_mode_from_path(path), "qwen35")

    def test_infer_from_path_base_dir(self):
        path = "/models/Qwen3.6-27B-exl3"
        self.assertEqual(infer_mode_from_path(path), "qwen35")


class BuildPromptAndStopsTests(unittest.TestCase):
    """Verify the catbench prompt gets wrapped in the chat template and
    that stop conditions pick up the template's turn-boundary tokens."""

    def _fake_qwen_tokenizer(self):
        """Minimal tokenizer stub that records the text it was asked to
        encode and reports <|im_end|> as a single-token id."""
        tok = MagicMock()
        tok.eos_token_id = 151643

        def _encode(text, add_bos=False, encode_special_tokens=False):
            tok.last_encoded = text
            tok.last_add_bos = add_bos
            tok.last_encode_special = encode_special_tokens
            out = MagicMock()
            out.shape = (1, len(text.split()))
            return out

        def _single_id(token):
            return {"<|im_end|>": 151645}.get(token)

        tok.encode.side_effect = _encode
        tok.single_id.side_effect = _single_id
        return tok

    def _fake_config(self):
        cfg = MagicMock()
        cfg.eos_token_id_list = [151643, 151645]
        return cfg

    def test_qwen35_wraps_prompt_in_chatml_frame(self):
        tok = self._fake_qwen_tokenizer()
        cfg = self._fake_config()
        _ids, stops, mode = build_prompt_and_stops("qwen35", tok, cfg)

        self.assertEqual(mode, "qwen35")
        # Prompt was framed as ChatML
        self.assertIn("<|im_start|>user", tok.last_encoded)
        self.assertIn(CATBENCH_PROMPT, tok.last_encoded)
        self.assertIn("<|im_end|>", tok.last_encoded)
        self.assertIn("<|im_start|>assistant", tok.last_encoded)
        # qwen35 template emits the "no-think" preamble when think=False
        self.assertIn("<think>", tok.last_encoded)
        self.assertIn("</think>", tok.last_encoded)
        # encode was called with encode_special_tokens=True
        self.assertTrue(tok.last_encode_special)
        # qwen35 doesn't prepend BOS
        self.assertFalse(tok.last_add_bos)

    def test_qwen35_stop_conditions_include_im_end(self):
        tok = self._fake_qwen_tokenizer()
        cfg = self._fake_config()
        _ids, stops, _ = build_prompt_and_stops("qwen35", tok, cfg)
        # Must include the <|im_end|> single-token id (151645), the literal
        # string fallback, and the EOS id.
        self.assertIn(151645, stops)
        self.assertIn("<|im_end|>", stops)
        self.assertIn(151643, stops)

    def test_unknown_mode_falls_back_to_chatml(self):
        tok = self._fake_qwen_tokenizer()
        cfg = self._fake_config()
        _ids, _stops, mode = build_prompt_and_stops("not-a-real-mode", tok, cfg)
        # Resolved mode string is returned as-passed, but the frame must
        # still be valid ChatML (the fallback path).
        self.assertEqual(mode, "not-a-real-mode")
        self.assertIn("<|im_start|>user", tok.last_encoded)


if __name__ == "__main__":
    unittest.main()
