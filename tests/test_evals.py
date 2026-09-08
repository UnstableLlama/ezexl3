"""Tests for the eval integration module (ezexl3/evals.py)."""

import csv
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
    format_eval_result,
    result_is_empty,
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
# Result display formatter tests
# ---------------------------------------------------------------------------

class TestFormatEvalResult:
    def test_perf_includes_both_metrics(self):
        result = {"perf_prefill_tps": "1200.00", "perf_gen_tps": "150.23"}
        out = format_eval_result("perf", result)
        assert "1200.00" in out
        assert "150.23" in out
        assert "N/A" not in out

    def test_perf_missing_values_show_na(self):
        out = format_eval_result("perf", {"perf_prefill_tps": "", "perf_gen_tps": ""})
        assert out.count("N/A") == 2

    def test_diversity(self):
        out = format_eval_result("diversity", {"diversity_score": "0.847362"})
        assert "0.847362" in out
        assert "diversity" in out.lower()

    def test_mmlu(self):
        out = format_eval_result("mmlu", {"mmlu_accuracy": "84.2%"})
        assert "84.2%" in out

    def test_humaneval(self):
        out = format_eval_result("humaneval", {"humaneval_pass": "0.652"})
        assert "0.652" in out
        assert "pass@1" in out

    def test_ifbench(self):
        out = format_eval_result("ifbench", {"ifbench_score": "0.823"})
        assert "0.823" in out

    def test_longctx(self):
        out = format_eval_result("longctx", {"longctx_score": "6/6"})
        assert "6/6" in out

    def test_every_registered_eval_has_formatter(self):
        """Every eval in the registry must produce a non-empty display string
        when given a populated result dict."""
        # One known value per DB column for each eval.
        samples = {
            "diversity": {"diversity_score": "0.5"},
            "humaneval": {"humaneval_pass": "0.5"},
            "ifbench": {"ifbench_score": "0.5"},
            "longctx": {"longctx_score": "3/6"},
            "mmlu": {"mmlu_accuracy": "50.0%"},
            "perf": {"perf_prefill_tps": "1000", "perf_gen_tps": "100"},
        }
        for name in EVAL_REGISTRY:
            assert format_eval_result(name, samples[name]), f"{name}: empty formatter"

    def test_unknown_eval_returns_empty(self):
        assert format_eval_result("nonexistent", {}) == ""


class TestResultIsEmpty:
    def test_fully_populated_is_not_empty(self):
        assert not result_is_empty("perf", {"perf_prefill_tps": "1000", "perf_gen_tps": "100"})

    def test_partially_populated_is_not_empty(self):
        # perf requires both columns for eval_has_result, but result_is_empty
        # only cares that at least one produced a value — partial extraction
        # is still worth surfacing to the user.
        assert not result_is_empty("perf", {"perf_prefill_tps": "1000", "perf_gen_tps": ""})

    def test_all_empty_is_empty(self):
        assert result_is_empty("perf", {"perf_prefill_tps": "", "perf_gen_tps": ""})

    def test_missing_keys_is_empty(self):
        assert result_is_empty("diversity", {})

    def test_whitespace_only_is_empty(self):
        assert result_is_empty("mmlu", {"mmlu_accuracy": "  "})

    def test_unknown_eval_is_empty(self):
        assert result_is_empty("nonexistent", {"anything": "1"})


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
        assert rows["4"]["PPL"] == "7.2"
        assert rows["4"]["MMLU"] == "84.2%"

    def test_export_csv_dynamic_eval_columns(self, tmp_path):
        """CSV should only include eval columns that have a populated row.

        This keeps the baseline output file clean (KL/PPL/GiB only) while
        still surfacing hidden-flag evals when they produce data.
        """
        db_path = str(tmp_path / "test.db")
        csv_path = str(tmp_path / "out.csv")
        upsert_row(db_path, weights="4", kl_div="0.5", mmlu_accuracy="84.2%")
        from ezexl3.measure_db import export_csv
        export_csv(db_path, csv_path)
        with open(csv_path) as f:
            header = f.readline()
        # Core columns: always present
        assert "weights" in header
        assert "KL Div" in header
        assert "PPL" in header
        assert "GiB" in header
        # Populated eval column: included
        assert "MMLU" in header
        # Unpopulated eval columns: omitted
        assert "Diversity" not in header
        assert "HumanEval" not in header
        assert "IFBench" not in header
        assert "LongCtx" not in header
        assert "Perf Prefill t/s" not in header
        assert "Perf Gen t/s" not in header

    def test_export_csv_core_only_when_no_evals(self, tmp_path):
        """With only KL/PPL/GiB populated the CSV is the classic 4 cols."""
        db_path = str(tmp_path / "test.db")
        csv_path = str(tmp_path / "out.csv")
        upsert_row(db_path, weights="4", kl_div="0.5", ppl="7.2", gib="3.1")
        from ezexl3.measure_db import export_csv
        export_csv(db_path, csv_path)
        with open(csv_path) as f:
            header = f.readline().strip()
        assert header == "weights,KL Div,PPL,GiB"


# ---------------------------------------------------------------------------
# Command builder tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI -> run_repo/run_measure_stage wiring tests
# ---------------------------------------------------------------------------

class TestEvalsCliWiring:
    def test_repo_passes_evals_to_run_repo(self):
        from unittest.mock import patch
        argv = ["repo", "-m", "/tmp/model", "-b", "4", "--no-readme", "-div", "-mmlu", "10"]
        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run_repo:
            from ezexl3.cli import main
            rc = main(argv)
        assert rc == 0
        kwargs = mock_run_repo.call_args.kwargs
        assert kwargs["evals"] is not None
        assert "diversity" in kwargs["evals"]
        assert kwargs["evals"]["diversity"] == 50  # const default
        assert "mmlu" in kwargs["evals"]
        assert kwargs["evals"]["mmlu"] == 10  # explicit value

    def test_measure_passes_evals_to_run_measure_stage(self):
        from unittest.mock import patch
        argv = ["measure", "-m", "/tmp/model", "-b", "4", "-div", "100"]
        with patch("ezexl3.repo.run_measure_stage", return_value=0) as mock_stage:
            from ezexl3.cli import main
            rc = main(argv)
        assert rc == 0
        kwargs = mock_stage.call_args.kwargs
        assert kwargs["evals"] is not None
        assert kwargs["evals"]["diversity"] == 100

    def test_repo_passes_skip_kl_ppl(self):
        from unittest.mock import patch
        argv = ["repo", "-m", "/tmp/model", "-b", "4", "--no-readme", "--no-kl", "--no-ppl"]
        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run_repo:
            from ezexl3.cli import main
            rc = main(argv)
        kwargs = mock_run_repo.call_args.kwargs
        assert kwargs["skip_kl"] is True
        assert kwargs["skip_ppl"] is True

    def test_no_evals_passes_none(self):
        from unittest.mock import patch
        argv = ["repo", "-m", "/tmp/model", "-b", "4", "--no-readme"]
        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run_repo:
            from ezexl3.cli import main
            rc = main(argv)
        kwargs = mock_run_repo.call_args.kwargs
        assert kwargs["evals"] is None


# ---------------------------------------------------------------------------
# Command builder tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# run_measure_stage done-handler integration tests
# ---------------------------------------------------------------------------

class _SyncProcess:
    """Process stand-in that runs target immediately in-thread.

    Lets us drive ``run_measure_stage``'s main results loop without spawning
    real subprocesses or touching GPUs.
    """

    def __init__(self, target, args):
        self._target = target
        self._args = args
        self.daemon = False

    def start(self):
        self._target(*self._args)

    def join(self, timeout=None):
        pass


class _FakeQueue:
    def __init__(self):
        self._items = []

    def put(self, item):
        self._items.append(item)

    def get(self):
        return self._items.pop(0)


def _run_stage_with_fake_worker(tmp_path, worker_fn, eval_name, eval_arg=32768):
    """Drive run_measure_stage with a stubbed worker and capture outputs.

    Returns (rc, captured_msgs, csv_path).
    """
    from ezexl3 import repo_measure
    from ezexl3 import measure_db

    model_dir = str(tmp_path)
    captured = []

    rc = repo_measure.run_measure_stage(
        model_dir=model_dir,
        bpws=["4"],
        devices=[0],
        write_logs=False,
        measure_args=[],
        evals={eval_name: eval_arg},
        skip_kl=True,
        skip_ppl=True,
        process_cls=_SyncProcess,
        queue_cls=_FakeQueue,
        worker_measure_fn=worker_fn,
        init_gpu_progress_fn=lambda *a, **kw: None,
        redraw_gpu_progress_fn=lambda *a, **kw: None,
        print_msg_with_progress_fn=lambda msg, *a, **kw: captured.append(msg),
        cleanup_gpu_progress_fn=lambda *a, **kw: None,
        clear_and_redraw_progress_fn=lambda *a, **kw: None,
        print_above_progress_fn=lambda *a, **kw: None,
        sleep_fn=lambda *a, **kw: None,
        wait_for_model_name_fn=lambda *a, **kw: None,
    )
    csv_path = measure_db.default_db_path(model_dir).replace(".db", ".csv")
    # default_csv_path is in ezexl3.measure, use it for accuracy
    from ezexl3.measure import default_csv_path
    csv_path = default_csv_path(model_dir)
    return rc, captured, csv_path


class TestRunMeasureStageEvalDoneHandler:
    def _make_worker(self, result_dict):
        """Build a fake worker that drains tasks and emits one done event
        per non-sentinel task with the given result_dict."""
        from ezexl3 import measure_db

        def fake_worker(model_dir, device, db_path, tasks, results, log_path, ppl_rows):
            while True:
                job = tasks.get()
                if job is None:
                    results.put(None)
                    return
                phase = job["phase"]
                task_label = job["label"]
                label = "bf16" if task_label == "base" else str(task_label)
                if result_dict:
                    measure_db.upsert_row(db_path, weights=label, **result_dict)
                results.put({
                    "event": "done", "device": device, "label": label,
                    "phase": phase, "row": dict(result_dict),
                })
        return fake_worker

    def test_perf_done_message_contains_metric_values(self, tmp_path):
        """The 'DONE' line must show the actual prefill/gen numbers, not 'PPL=N/A'."""
        worker = self._make_worker({"perf_prefill_tps": "1234.56", "perf_gen_tps": "56.78"})
        rc, captured, _ = _run_stage_with_fake_worker(tmp_path, worker, "perf")
        assert rc == 0
        done_msgs = [m for m in captured if "DONE" in m]
        assert done_msgs, f"No DONE messages captured. All msgs: {captured}"
        # Every DONE msg for a perf task must contain the prefill and gen values.
        assert any("1234.56" in m for m in done_msgs), f"prefill missing: {done_msgs}"
        assert any("56.78" in m for m in done_msgs), f"gen missing: {done_msgs}"
        # None of them should say "PPL=N/A" — that was the old dragnet bug.
        assert not any("PPL=N/A" in m for m in done_msgs), f"PPL=N/A leaked: {done_msgs}"
        assert not any("PPL=" in m for m in done_msgs), f"PPL= leaked: {done_msgs}"

    def test_perf_done_writes_to_csv(self, tmp_path):
        """CSV export runs after a perf task completes (not just kl/ppl)."""
        worker = self._make_worker({"perf_prefill_tps": "1234.56", "perf_gen_tps": "56.78"})
        rc, _, csv_path = _run_stage_with_fake_worker(tmp_path, worker, "perf")
        assert rc == 0
        assert os.path.isfile(csv_path), f"CSV not exported at {csv_path}"
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        by_label = {r["weights"]: r for r in rows}
        assert "4" in by_label, f"'4' row missing from CSV. Rows: {rows}"
        assert by_label["4"]["Perf Prefill t/s"] == "1234.56"
        assert by_label["4"]["Perf Gen t/s"] == "56.78"

    def test_diversity_done_message_contains_score(self, tmp_path):
        worker = self._make_worker({"diversity_score": "0.847362"})
        rc, captured, _ = _run_stage_with_fake_worker(tmp_path, worker, "diversity", eval_arg=50)
        assert rc == 0
        done_msgs = [m for m in captured if "DONE" in m]
        assert any("0.847362" in m for m in done_msgs), f"diversity score missing: {done_msgs}"

    def test_empty_result_surfaces_as_warning(self, tmp_path):
        """A worker 'done' with an all-empty result dict must surface a warning
        and count as a failure, not silently pretend the eval succeeded."""
        worker = self._make_worker({"perf_prefill_tps": "", "perf_gen_tps": ""})
        rc, captured, _ = _run_stage_with_fake_worker(tmp_path, worker, "perf")
        # Non-zero return code because result_is_empty flagged a failure.
        assert rc != 0, f"Expected failure rc for empty result, got {rc}. Msgs: {captured}"
        warning_msgs = [m for m in captured if "no results extracted" in m]
        assert warning_msgs, f"Expected warning msg, got: {captured}"


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


class TestPerfRunnerInvocation:
    """Perf eval runs through the heartbeat-emitting wrapper, not the
    raw vendored script."""

    def test_perf_cmd_uses_runner_wrapper(self, tmp_path):
        cmd = build_eval_cmd("perf", str(tmp_path), 0, str(tmp_path), "4", 16384)
        # Should invoke our wrapper module, not the vendored script directly.
        assert "-m" in cmd
        assert "ezexl3.perf_runner" in cmd
        # Vendored eval_perf.py should NOT be on the command line for perf.
        assert not any("eval_perf.py" in c for c in cmd)

    def test_perf_cmd_sets_cache_size_to_max_length(self, tmp_path):
        # Upstream's measure_generate caps past_len at max_length - 256 and
        # reserves length + 256 in batch_shape, so cache = max_length is
        # sufficient for the final gen iteration.
        cmd = build_eval_cmd("perf", str(tmp_path), 0, str(tmp_path), "4", 16384)
        assert "-max_length" in cmd
        assert cmd[cmd.index("-max_length") + 1] == "16384"
        assert "-cs" in cmd
        assert cmd[cmd.index("-cs") + 1] == "16384"

    def test_perf_cmd_default_max_length_matches_cache(self, tmp_path):
        # eval_arg=0 falls back to the 32768 default; -cs tracks 1:1.
        cmd = build_eval_cmd("perf", str(tmp_path), 0, str(tmp_path), "4", 0)
        assert cmd[cmd.index("-max_length") + 1] == "32768"
        assert cmd[cmd.index("-cs") + 1] == "32768"

    def test_other_evals_still_use_vendored_script(self, tmp_path):
        # mmlu (and friends) still invoke the vendored script directly.
        cmd = build_eval_cmd("mmlu", str(tmp_path), 0, str(tmp_path), "4", 5)
        assert any("eval_mmlu.py" in c for c in cmd)

    def test_perf_heartbeat_parser_recognises_marker(self):
        # Inner-loop heartbeat lines from perf_runner are forwarded to the UI.
        out = _parse_perf_progress(
            "PERF_HEARTBEAT gen length=32768 50/100 (8.50 t/s)"
        )
        assert out is not None
        assert "gen length=32768" in out
        assert "50/100" in out
        assert "8.50 t/s" in out


class TestRunEvalSubprocessPartialSalvage:
    """When an eval subprocess crashes mid-run but has already printed
    parseable output, run_eval_subprocess should salvage the partial output
    instead of raising and discarding everything."""

    def _make_q(self):
        from queue import Queue as TQueue
        return TQueue()

    def test_partial_perf_output_is_salvaged_on_nonzero_exit(self):
        """Simulates eval_perf.py printing prefill + some generation rows
        and then crashing (non-zero exit). Expect: no RuntimeError,
        returned output contains the parseable rows."""
        import sys as _sys
        from ezexl3.evals import run_eval_subprocess
        script = (
            "import sys; "
            "print('Prefill:'); "
            "print('Length    256:     1000.00 tokens/s'); "
            "print('Length    512:      900.00 tokens/s'); "
            "print('Generation'); "
            "print('Context      0:      120.00 tokens/s'); "
            "print('Context    256:      110.00 tokens/s'); "
            "sys.stdout.flush(); "
            "sys.exit(1)"
        )
        cmd = [_sys.executable, "-c", script]
        q = self._make_q()
        out = run_eval_subprocess(
            cmd, device=0, results=q,
            phase_label="4 PERF", eval_name="perf",
        )
        assert "1000.00 tokens/s" in out
        assert "120.00 tokens/s" in out

    def test_no_parseable_output_still_raises(self):
        """When the subprocess exits nonzero and the extractor finds
        nothing, run_eval_subprocess must still raise so the caller
        reports a full failure."""
        import sys as _sys
        from ezexl3.evals import run_eval_subprocess
        script = "import sys; print('boom'); sys.exit(1)"
        cmd = [_sys.executable, "-c", script]
        q = self._make_q()
        with pytest.raises(RuntimeError, match="failed with exit code"):
            run_eval_subprocess(
                cmd, device=0, results=q,
                phase_label="4 PERF", eval_name="perf",
            )
