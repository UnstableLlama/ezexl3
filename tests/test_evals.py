"""Tests for the eval integration module (ezexl3/evals.py)."""

import os
import sqlite3
import tempfile

import pytest

from ezexl3.evals import (
    EVAL_REGISTRY,
    EVAL_QUEUE_ORDER,
    PROGRESS_PARSERS,
    RESULT_EXTRACTORS,
    eval_has_result,
    build_eval_cmd,
    detect_prompt_format,
    _parse_diversity_progress,
    _parse_humaneval_progress,
    _parse_ifbench_progress,
    _parse_longctx_progress,
    _parse_mmlu_progress,
    _parse_perf_progress,
    _extract_diversity_result,
    _extract_humaneval_result,
    _extract_ifbench_result,
    _extract_longctx_result,
    _extract_mmlu_result,
    _extract_perf_result,
)
from ezexl3.measure_db import upsert_row, default_db_path, _EVAL_COLUMNS


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestEvalRegistry:
    def test_all_six_evals_registered(self):
        expected = {"diversity", "humaneval", "ifbench", "longctx", "mmlu", "perf"}
        assert set(EVAL_REGISTRY.keys()) == expected

    def test_queue_order_covers_all(self):
        assert set(EVAL_QUEUE_ORDER) == set(EVAL_REGISTRY.keys())

    def test_all_have_progress_parser(self):
        for name in EVAL_REGISTRY:
            assert name in PROGRESS_PARSERS, f"Missing progress parser for {name}"

    def test_all_have_result_extractor(self):
        for name in EVAL_REGISTRY:
            assert name in RESULT_EXTRACTORS, f"Missing result extractor for {name}"

    def test_db_columns_are_valid(self):
        for name, eval_def in EVAL_REGISTRY.items():
            for col in eval_def.db_columns:
                assert col in _EVAL_COLUMNS, f"{name}: db column {col} not in _EVAL_COLUMNS"

    def test_each_has_phase_label(self):
        for name, eval_def in EVAL_REGISTRY.items():
            assert eval_def.phase_label, f"{name}: missing phase_label"
            assert len(eval_def.phase_label) <= 6, f"{name}: phase_label too long"


# ---------------------------------------------------------------------------
# Progress parser tests
# ---------------------------------------------------------------------------

class TestDiversityProgress:
    def test_percentage(self):
        assert _parse_diversity_progress("Inference  42%  0:01:30") == "gen 42%"

    def test_extraction_phase(self):
        assert _parse_diversity_progress("Extracting variables  80%") == "extract 80%"

    def test_mean_result(self):
        assert _parse_diversity_progress("mean                  0.847362") == "complete"

    def test_irrelevant_line(self):
        assert _parse_diversity_progress("Loading model...") is None

    def test_empty(self):
        assert _parse_diversity_progress("") is None


class TestHumanEvalProgress:
    def test_problem_sample(self):
        result = _parse_humaneval_progress(" ** Problem 12, sample 3 / 200")
        assert result == "p12 s3/200"

    def test_percentage(self):
        result = _parse_humaneval_progress("Creating sample jobs  65%")
        assert result == "setup 65%"

    def test_generation_percentage(self):
        result = _parse_humaneval_progress("Generating samples  42%")
        assert result == "gen 42%"

    def test_saving(self):
        assert _parse_humaneval_progress(" -- Saving: output.jsonl") == "saving"


class TestIFBenchProgress:
    def test_pending_active(self):
        result = _parse_ifbench_progress(" -- pending:    2    active    4   847 tokens/s")
        assert "pend 2" in result
        assert "847" in result

    def test_percentage(self):
        assert _parse_ifbench_progress("Prompts  75%") == "75%"

    def test_responses_written(self):
        assert _parse_ifbench_progress(" -- Responses written to out.jsonl") == "complete"


class TestLongCtxProgress:
    def test_test_header(self):
        result = _parse_longctx_progress("\x1b[32mSUMMARY TEST\x1b[0m")
        assert "summary test" in result

    def test_percentage(self):
        assert _parse_longctx_progress("Inference  60%") == "inference 60%"


class TestMMLUProgress:
    def test_subject_result(self):
        result = _parse_mmlu_progress("biology:                                  125/  150 =  83.33% correct, ( 82.41% prob.)")
        assert "biology" in result
        assert "83.33%" in result

    def test_preprompts(self):
        assert _parse_mmlu_progress("Preprompts  30%") == "preprompts 30%"

    def test_testing(self):
        assert _parse_mmlu_progress("Testing  85%") == "testing 85%"


class TestPerfProgress:
    def test_prefill_result(self):
        result = _parse_perf_progress("Length     256:       1847.35 tokens/s")
        assert "prefill" in result
        assert "1847.35" in result

    def test_gen_result(self):
        result = _parse_perf_progress("Context   4096:        812.47 tokens/s")
        assert "gen" in result
        assert "812.47" in result

    def test_warmup(self):
        assert _parse_perf_progress("Warmup  50%") == "warmup 50%"


# ---------------------------------------------------------------------------
# Result extractor tests
# ---------------------------------------------------------------------------

class TestDiversityExtractor:
    def test_valid(self):
        output = "some stuff\nmean                  0.847362\nmore stuff"
        assert _extract_diversity_result(output) == {"diversity_score": "0.847362"}

    def test_missing(self):
        assert _extract_diversity_result("no result here") == {"diversity_score": ""}


class TestHumanEvalExtractor:
    def test_pass_at_1(self):
        output = "pass@1 (200 samples): 0.652"
        assert _extract_humaneval_result(output) == {"humaneval_pass": "0.652"}

    def test_saving_fallback(self):
        output = " -- Saving: output.jsonl"
        assert _extract_humaneval_result(output) == {"humaneval_pass": "done"}


class TestIFBenchExtractor:
    def test_score(self):
        output = "Overall accuracy: 0.823"
        assert _extract_ifbench_result(output) == {"ifbench_score": "0.823"}

    def test_written_fallback(self):
        output = " -- Responses written to out.jsonl"
        assert _extract_ifbench_result(output) == {"ifbench_score": "done"}


class TestLongCtxExtractor:
    def test_all_tests_found(self):
        output = (
            "SUMMARY TEST\nsome output\n"
            "FRENCH TEST\nsome output\n"
            "ZOOMER TEST\nsome output\n"
            "Q&A TEST\nsome output\n"
            "CORRUPTION TEST\nsome output\n"
            "NAME EXTRACTION TEST\nsome output\n"
        )
        assert _extract_longctx_result(output) == {"longctx_score": "6/6"}

    def test_partial(self):
        output = "SUMMARY TEST\nFRENCH TEST\nZOOMER TEST\n"
        assert _extract_longctx_result(output) == {"longctx_score": "3/6"}


class TestMMLUExtractor:
    def test_full_mode(self):
        output = "all subjects:                                842/ 1000 =  84.20% correct, ( 82.41% prob.)"
        assert _extract_mmlu_result(output) == {"mmlu_accuracy": "84.20%"}

    def test_random_mode(self):
        output = "all subjects, 1000 random samples:        842/ 1000 =  84.20% +/-   2.90%"
        assert _extract_mmlu_result(output) == {"mmlu_accuracy": "84.20%"}


class TestPerfExtractor:
    def test_both_metrics(self):
        output = (
            "Prefill:\n"
            "Length     256:       1847.35 tokens/s\n"
            "Length    4096:       1200.00 tokens/s\n"
            "Generation:\n"
            "Context      0:        150.23 tokens/s\n"
            "Context   4096:        120.00 tokens/s\n"
        )
        result = _extract_perf_result(output)
        assert result["perf_prefill_tps"] == "1200.00"  # last prefill
        assert result["perf_gen_tps"] == "150.23"  # first gen (context 0)


# ---------------------------------------------------------------------------
# Checkpoint logic tests
# ---------------------------------------------------------------------------

class TestEvalHasResult:
    def test_no_result(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        upsert_row(db_path, weights="4", kl_div="0.5")
        assert not eval_has_result(db_path, "4", "mmlu")

    def test_has_result(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        upsert_row(db_path, weights="4", mmlu_accuracy="84.2%")
        assert eval_has_result(db_path, "4", "mmlu")

    def test_perf_needs_both_columns(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        upsert_row(db_path, weights="4", perf_prefill_tps="1200")
        assert not eval_has_result(db_path, "4", "perf")

        upsert_row(db_path, weights="4", perf_gen_tps="150")
        assert eval_has_result(db_path, "4", "perf")


# ---------------------------------------------------------------------------
# CLI flag tests
# ---------------------------------------------------------------------------

class TestCLIEvalFlags:
    def test_repo_has_eval_flags(self):
        from ezexl3.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "repo", "-m", "/tmp/model", "-b", "4",
            "-mmlu", "-perf", "-div", "100",
        ])
        assert args.mmlu == 5  # const default
        assert args.perf == 32768  # const default
        assert args.diversity == 100  # explicit value

    def test_measure_has_eval_flags(self):
        from ezexl3.cli import build_parser
        parser = build_parser()
        args = parser.parse_args([
            "measure", "-m", "/tmp/model", "-b", "4",
            "-lctx", "-he",
        ])
        assert args.longctx == 1  # const default
        assert args.humaneval == 200  # const default

    def test_eval_flags_default_to_zero(self):
        from ezexl3.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["repo", "-m", "/tmp/model", "-b", "4"])
        assert args.mmlu == 0
        assert args.perf == 0
        assert args.diversity == 0
        assert args.humaneval == 0
        assert args.ifbench == 0
        assert args.longctx == 0

    def test_no_kl_no_ppl_flags(self):
        from ezexl3.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["repo", "-m", "/tmp/model", "-b", "4", "--no-kl", "--no-ppl"])
        assert args.no_kl is True
        assert args.no_ppl is True

    def test_no_kl_no_ppl_default_off(self):
        from ezexl3.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["repo", "-m", "/tmp/model", "-b", "4"])
        assert args.no_kl is False
        assert args.no_ppl is False

    def test_measure_no_kl_no_ppl_flags(self):
        from ezexl3.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["measure", "-m", "/tmp/model", "-b", "4", "--no-kl", "--no-ppl"])
        assert args.no_kl is True
        assert args.no_ppl is True


# ---------------------------------------------------------------------------
# Database column tests
# ---------------------------------------------------------------------------

class TestMeasureDBEvalColumns:
    def test_upsert_eval_column(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        upsert_row(db_path, weights="4", mmlu_accuracy="84.2%")
        from ezexl3.measure_db import read_all_rows
        rows = read_all_rows(db_path)
        assert rows["4"]["MMLU"] == "84.2%"

    def test_eval_upsert_doesnt_blank_kl(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        upsert_row(db_path, weights="4", kl_div="0.5", ppl="7.2")
        upsert_row(db_path, weights="4", mmlu_accuracy="84.2%")
        from ezexl3.measure_db import read_all_rows
        rows = read_all_rows(db_path)
        assert rows["4"]["KL Div"] == "0.5"
        assert rows["4"]["PPL r-100"] == "7.2"
        assert rows["4"]["MMLU"] == "84.2%"

    def test_export_csv_includes_eval_columns(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        csv_path = str(tmp_path / "out.csv")
        upsert_row(db_path, weights="4", kl_div="0.5", mmlu_accuracy="84.2%")
        from ezexl3.measure_db import export_csv
        export_csv(db_path, csv_path)
        with open(csv_path) as f:
            header = f.readline()
        assert "MMLU" in header
        assert "Diversity" in header
        assert "Perf Prefill t/s" in header


# ---------------------------------------------------------------------------
# Command builder tests
# ---------------------------------------------------------------------------

class TestBuildEvalCmd:
    def test_mmlu_cmd(self, tmp_path):
        cmd = build_eval_cmd("mmlu", str(tmp_path), 0, str(tmp_path), "4", 10)
        assert "-fs" in cmd
        idx = cmd.index("-fs")
        assert cmd[idx + 1] == "10"
        # Verify it uses the vendored script path
        assert "vendor" in cmd[1]
        assert "eval_mmlu.py" in cmd[1]
        # Device is set via CUDA_VISIBLE_DEVICES, not -d
        assert "-d" not in cmd

    def test_humaneval_creates_output_dir(self, tmp_path):
        base = str(tmp_path)
        cmd = build_eval_cmd("humaneval", base, 0, base, "4", 50)
        out_dir = os.path.join(base, "evals", "humaneval")
        assert os.path.isdir(out_dir)
        assert "-spt" in cmd
        idx = cmd.index("-spt")
        assert cmd[idx + 1] == "50"

    def test_vendored_scripts_exist(self):
        """All vendored eval scripts must exist in the package."""
        from ezexl3.evals import _vendor_script
        for name in EVAL_REGISTRY:
            path = _vendor_script(name)
            assert os.path.isfile(path), f"Vendored script missing: {path}"
