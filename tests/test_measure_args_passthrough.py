import csv
import os
import tempfile
import unittest
from unittest.mock import patch, call, MagicMock

from ezexl3 import cli
from ezexl3 import repo
from ezexl3 import measure_db


class MeasureArgsPassthroughTests(unittest.TestCase):
    def test_parse_measure_args_defaults(self):
        rows, devices = repo._parse_measure_args([], [0, 1])
        self.assertEqual(rows, 100)
        self.assertEqual(devices, [0, 1])

    def test_parse_measure_args_rows_and_devices(self):
        rows, devices = repo._parse_measure_args(["-r", "200", "-d", "2,3"], [0])
        self.assertEqual(rows, 200)
        self.assertEqual(devices, [2, 3])

    def test_parse_measure_args_rejects_non_positive_rows(self):
        with self.assertRaises(ValueError):
            repo._parse_measure_args(["-r", "0"], [0])

    def test_parse_measure_args_rejects_unknown_flag(self):
        with self.assertRaises(ValueError):
            repo._parse_measure_args(["--foo", "bar"], [0])

    def test_repo_command_passes_measure_args_to_run_repo(self):
        argv = [
            "repo",
            "-m",
            "/tmp/model",
            "-b",
            "2",
            "--no-readme",
            "--measure-args",
            "--",
            "-r",
            "150",
            "-d",
            "1",
        ]

        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run_repo:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        kwargs = mock_run_repo.call_args.kwargs
        self.assertEqual(kwargs["measure_args"], ["-r", "150", "-d", "1"])
        self.assertEqual(kwargs["optimized_measure_layers"], 2)

    def test_repo_command_passes_layers_to_run_repo(self):
        argv = [
            "repo",
            "-m",
            "/tmp/model",
            "-b",
            "2",
            "--no-readme",
            "-l",
            "1",
        ]

        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run_repo:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        kwargs = mock_run_repo.call_args.kwargs
        self.assertEqual(kwargs["optimized_measure_layers"], 1)

    def test_run_measure_stage_rejects_empty_devices_after_passthrough(self):
        with self.assertRaises(ValueError):
            repo.run_measure_stage(
                model_dir="/tmp/model",
                bpws=["2"],
                devices=[0],
                write_logs=False,
                measure_args=["-d", ""],
            )

    def test_measure_command_passes_measure_args_to_run_measure_stage(self):
        argv = [
            "measure",
            "-m",
            "/tmp/model",
            "-b",
            "2",
            "--measure-args",
            "--",
            "-r",
            "180",
            "-d",
            "2",
        ]

        with patch("ezexl3.repo.run_measure_stage", return_value=0) as mock_stage:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        kwargs = mock_stage.call_args.kwargs
        self.assertEqual(kwargs["measure_args"], ["-r", "180", "-d", "2"])


class MeasureCheckpointingTests(unittest.TestCase):
    def test_task_to_csv_label_maps_base_to_bf16(self):
        self.assertEqual(repo._task_to_csv_label("base"), "bf16")
        self.assertEqual(repo._task_to_csv_label("4"), "4")

    def test_filter_measure_tasks_for_checkpoint_skips_existing_rows(self):
        requested = ["2", "3", "base"]
        existing = {"3", "bf16"}
        self.assertEqual(repo._filter_measure_tasks_for_checkpoint(requested, existing), ["2"])

    def test_run_measure_stage_returns_early_when_all_rows_measured(self):
        full_rows = {
            "2": {"weights": "2", "KL Div": "0.1", "PPL": "11.0", "GiB": "4.2"},
            "bf16": {"weights": "bf16", "KL Div": "0.0", "PPL": "10.0", "GiB": "12.3"},
        }
        with patch("ezexl3.repo._read_db_rows", return_value=full_rows), \
             patch("ezexl3.repo.migrate_csv_to_db"), \
             patch("ezexl3.repo.Process") as mock_process:
            rc = repo.run_measure_stage(
                model_dir="/tmp/model",
                bpws=["2"],
                devices=[0],
                write_logs=False,
                measure_args=[],
            )

        self.assertEqual(rc, 0)
        mock_process.assert_not_called()

    def test_upsert_preserves_existing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "ModelMeasured.db")
            csv_path = os.path.join(tmp, "ModelMeasured.csv")

            measure_db.upsert_row(db_path, weights="bf16", kl_div="0.0", ppl="10.0", gib="12.3")
            measure_db.upsert_row(db_path, weights="2", kl_div="0.1", ppl="11.0", gib="4.2")

            measure_db.export_csv(db_path, csv_path)
            with open(csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))

        labels = [r["weights"] for r in rows]
        # Sorted numerically: 2 first, then bf16 (non-numeric sorts last)
        self.assertEqual(labels, ["2", "bf16"])

    def test_upsert_combines_partial_rows(self):
        """KL-only and PPL-only upserts for the same label merge into one complete row."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "ModelMeasured.db")
            csv_path = os.path.join(tmp, "ModelMeasured.csv")

            # Simulate GPU 0 writing KL only
            measure_db.upsert_row(db_path, weights="4", kl_div="0.05", gib="6.0")
            # Simulate GPU 1 writing PPL only
            measure_db.upsert_row(db_path, weights="4", ppl="9.8", gib="6.0")

            measure_db.export_csv(db_path, csv_path)
            with open(csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["weights"], "4")
        self.assertEqual(rows[0]["KL Div"], "0.05")
        self.assertEqual(rows[0]["PPL"], "9.8")

    def test_upsert_overwrites_with_newer_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "ModelMeasured.db")

            measure_db.upsert_row(db_path, weights="2", kl_div="9.9", ppl="99.0", gib="4.0")
            measure_db.upsert_row(db_path, weights="2", kl_div="0.2", ppl="12.0", gib="4.1")

            rows = measure_db.read_all_rows(db_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows["2"]["KL Div"], "0.2")
        self.assertEqual(rows["2"]["PPL"], "12.0")

    def test_migrate_csv_to_db(self):
        """Legacy CSV data is imported into the database."""
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "ModelMeasured.csv")
            db_path = os.path.join(tmp, "ModelMeasured.db")
            # Pre-qbench header: the perplexity column was "PPL r-100".
            fields = ["weights", "KL Div", "PPL r-100", "GiB"]

            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow({"weights": "bf16", "KL Div": "0.0", "PPL r-100": "10.0", "GiB": "12.3"})
                w.writerow({"weights": "3", "KL Div": "0.08", "PPL r-100": "9.5", "GiB": "5.1"})

            count = measure_db.migrate_csv_to_db(csv_path, db_path)
            self.assertEqual(count, 2)

            rows = measure_db.read_all_rows(db_path)
            self.assertIn("bf16", rows)
            self.assertIn("3", rows)
            self.assertEqual(rows["3"]["KL Div"], "0.08")


class InterleavedPipelineTests(unittest.TestCase):
    """Tests for the verify=True (default) interleaved quant→measure pipeline."""

    def _base_patches(self):
        """Common patches for run_repo tests."""
        return {
            "quant_run_one": patch("ezexl3.repo._run_quant_one_isolated", return_value=True),
            "run_measure_single_bpw": patch("ezexl3.repo.run_measure_single_bpw", return_value=0),
            "run_measure_stage": patch("ezexl3.repo.run_measure_stage", return_value=0),
            "_run_optimized_opt_stage": patch("ezexl3.repo._run_optimized_opt_stage"),
            "_init_measure_db": patch("ezexl3.repo._init_measure_db", return_value=("/tmp/db.sqlite", "/tmp/out.csv")),
            "export_csv": patch("ezexl3.repo.export_csv"),
        }

    def test_verify_true_quants_and_verifies_per_bpw(self):
        """With verify=True, each BPW is quantized then immediately verified."""
        patches = self._base_patches()
        mocks = {k: p.start() for k, p in patches.items()}
        try:
            rc = repo.run_repo(
                model_dir="/tmp/model",
                bpws=["2", "4"],
                devices=[0, 1],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_quant=True,
                do_measure=True,
                do_readme=False,
                verify=True,
            )
        finally:
            for p in patches.values():
                p.stop()

        self.assertEqual(rc, 0)
        # quant_run_one called once per integer BPW
        quant_calls = mocks["quant_run_one"].call_args_list
        quant_bpws = [c.args[1] for c in quant_calls]
        self.assertIn("2", quant_bpws)
        self.assertIn("4", quant_bpws)

        # run_measure_single_bpw called once per BPW (interleaved verification)
        measure_calls = mocks["run_measure_single_bpw"].call_args_list
        measured_bpws = [c.kwargs["bpw"] for c in measure_calls]
        self.assertIn("2", measured_bpws)
        self.assertIn("4", measured_bpws)

        # run_measure_stage NOT called (no catbench)
        mocks["run_measure_stage"].assert_not_called()

    def test_verify_true_halts_on_quant_failure(self):
        """Pipeline halts immediately if a quantization fails."""
        patches = self._base_patches()
        mocks = {k: p.start() for k, p in patches.items()}
        mocks["quant_run_one"].return_value = False  # quant fails
        try:
            rc = repo.run_repo(
                model_dir="/tmp/model",
                bpws=["2", "4"],
                devices=[0],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_quant=True,
                do_measure=True,
                do_readme=False,
                verify=True,
            )
        finally:
            for p in patches.values():
                p.stop()

        self.assertEqual(rc, 1)
        # Only one quant attempted before halting
        self.assertEqual(mocks["quant_run_one"].call_count, 1)
        # Verification never reached
        mocks["run_measure_single_bpw"].assert_not_called()

    def test_verify_true_halts_on_verification_failure(self):
        """Pipeline halts if the *first* per-BPW verification fails.

        This is the canary: if the very first verify is broken, the user
        almost certainly has a configuration/environment problem and the
        run should stop so they can fix it.
        """
        patches = self._base_patches()
        mocks = {k: p.start() for k, p in patches.items()}
        mocks["run_measure_single_bpw"].return_value = 1  # verify fails
        try:
            rc = repo.run_repo(
                model_dir="/tmp/model",
                bpws=["2", "4"],
                devices=[0],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_quant=True,
                do_measure=True,
                do_readme=False,
                verify=True,
            )
        finally:
            for p in patches.values():
                p.stop()

        self.assertEqual(rc, 1)
        # First quant succeeded but verification failed, so only 1 quant attempted
        self.assertEqual(mocks["quant_run_one"].call_count, 1)
        # Verification was called once and failed
        self.assertEqual(mocks["run_measure_single_bpw"].call_count, 1)

    def test_verify_continues_after_first_success(self):
        """If the first verify passes, later verify failures are non-fatal.

        The user has already seen the pipeline work once, so a later flake
        (e.g. transient VRAM pressure on a single BPW) should not trash the
        entire multi-BPW run.
        """
        patches = self._base_patches()
        mocks = {k: p.start() for k, p in patches.items()}
        # First verify passes, second fails, third passes
        mocks["run_measure_single_bpw"].side_effect = [0, 1, 0]
        try:
            rc = repo.run_repo(
                model_dir="/tmp/model",
                bpws=["2", "4", "6"],
                devices=[0],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_quant=True,
                do_measure=True,
                do_readme=False,
                verify=True,
            )
        finally:
            for p in patches.values():
                p.stop()

        # Run completes successfully despite the second verify failing
        self.assertEqual(rc, 0)
        # All three BPWs were quantized and verified
        self.assertEqual(mocks["quant_run_one"].call_count, 3)
        self.assertEqual(mocks["run_measure_single_bpw"].call_count, 3)

    def test_verify_continues_after_first_checkpoint_skip(self):
        """A checkpoint-skipped first verify (rc=0) still counts as 'first
        passed', so a subsequent failure is non-fatal."""
        patches = self._base_patches()
        mocks = {k: p.start() for k, p in patches.items()}
        # First BPW is already checkpointed (rc=0, no-op), second fails
        mocks["run_measure_single_bpw"].side_effect = [0, 1]
        try:
            rc = repo.run_repo(
                model_dir="/tmp/model",
                bpws=["2", "4"],
                devices=[0],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_quant=True,
                do_measure=True,
                do_readme=False,
                verify=True,
            )
        finally:
            for p in patches.values():
                p.stop()

        self.assertEqual(rc, 0)
        self.assertEqual(mocks["quant_run_one"].call_count, 2)
        self.assertEqual(mocks["run_measure_single_bpw"].call_count, 2)

    def test_verify_false_uses_legacy_pipeline(self):
        """With verify=False (--no-verify), uses batch quant then batch measure."""
        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, \
             patch("ezexl3.repo.run_measure_stage", return_value=0) as mock_measure:
            rc = repo.run_repo(
                model_dir="/tmp/model",
                bpws=["2", "4"],
                devices=[0],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_quant=True,
                do_measure=True,
                do_readme=False,
                verify=False,
            )

        self.assertEqual(rc, 0)
        mock_quant.assert_called_once()
        mock_measure.assert_called_once()

    def test_no_verify_cli_flag_passes_verify_false(self):
        """--no-verify flag results in verify=False passed to run_repo."""
        argv = [
            "repo", "-m", "/tmp/model", "-b", "2",
            "--no-readme", "--no-verify",
        ]
        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run_repo:
            cli.main(argv)

        self.assertFalse(mock_run_repo.call_args.kwargs["verify"])

    def test_verify_default_is_true(self):
        """Without --no-verify, verify defaults to True."""
        argv = [
            "repo", "-m", "/tmp/model", "-b", "2",
            "--no-readme",
        ]
        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run_repo:
            cli.main(argv)

        self.assertTrue(mock_run_repo.call_args.kwargs["verify"])

    def test_verify_with_fractional_bpws_no_opt(self):
        """Fractional BPWs without -opt are quantized directly (no optimization)."""
        patches = self._base_patches()
        mocks = {k: p.start() for k, p in patches.items()}
        try:
            rc = repo.run_repo(
                model_dir="/tmp/model",
                bpws=["3.5"],
                devices=[0, 1],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_quant=True,
                do_measure=True,
                do_readme=False,
                verify=True,
            )
        finally:
            for p in patches.values():
                p.stop()

        self.assertEqual(rc, 0)
        mocks["_run_optimized_opt_stage"].assert_not_called()
        measured_bpws = [c.kwargs["bpw"] for c in mocks["run_measure_single_bpw"].call_args_list]
        self.assertIn("3.5", measured_bpws)

    def test_verify_with_optimized_bpws(self):
        """Fractional BPWs with -opt are quantized, optimized, then verified."""
        patches = self._base_patches()
        mocks = {k: p.start() for k, p in patches.items()}
        try:
            rc = repo.run_repo(
                model_dir="/tmp/model",
                bpws=["3.5"],
                devices=[0, 1],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_quant=True,
                do_measure=True,
                do_readme=False,
                verify=True,
                opt_bpws={"3.5"},
            )
        finally:
            for p in patches.values():
                p.stop()

        self.assertEqual(rc, 0)
        mocks["_run_optimized_opt_stage"].assert_called_once()
        measured_bpws = [c.kwargs["bpw"] for c in mocks["run_measure_single_bpw"].call_args_list]
        self.assertIn("3.5", measured_bpws)

    def test_verify_passes_all_gpus_to_single_bpw_measure(self):
        """run_measure_single_bpw receives all devices for multi-GPU measurement."""
        patches = self._base_patches()
        mocks = {k: p.start() for k, p in patches.items()}
        try:
            repo.run_repo(
                model_dir="/tmp/model",
                bpws=["2"],
                devices=[0, 1, 2, 3],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_quant=True,
                do_measure=True,
                do_readme=False,
                verify=True,
            )
        finally:
            for p in patches.values():
                p.stop()

        # All 4 GPUs passed to per-BPW verification
        self.assertEqual(mocks["run_measure_single_bpw"].call_args.kwargs["devices"], [0, 1, 2, 3])

    def test_verify_measure_stage_receives_catbench_n(self):
        """catbench_n is forwarded to run_measure_stage in interleaved mode."""
        patches = self._base_patches()
        mocks = {k: p.start() for k, p in patches.items()}
        try:
            repo.run_repo(
                model_dir="/tmp/model",
                bpws=["2"],
                devices=[0],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_quant=True,
                do_measure=True,
                do_readme=False,
                verify=True,
                catbench_n=5,
            )
        finally:
            for p in patches.values():
                p.stop()

        # With catbench_n > 0, run_measure_stage is called for catbench
        mocks["run_measure_stage"].assert_called_once()
        self.assertEqual(mocks["run_measure_stage"].call_args.kwargs["catbench_n"], 5)


class GiBGapFillTests(unittest.TestCase):
    """Tests for GiB gap detection and filling in run_measure_stage."""

    def test_gib_gaps_filled_before_measurement(self):
        """Missing GiB values are filled from filesystem before GPU work starts."""
        full_rows = {
            "2": {"weights": "2", "KL Div": "0.1", "PPL": "11.0", "GiB": ""},
            "bf16": {"weights": "bf16", "KL Div": "0.0", "PPL": "10.0", "GiB": "12.3"},
        }
        with patch("ezexl3.repo._read_db_rows", return_value=full_rows), \
             patch("ezexl3.repo.migrate_csv_to_db"), \
             patch("ezexl3.repo.file_size_gib", return_value=4.2) as mock_gib, \
             patch("ezexl3.repo.upsert_row") as mock_upsert, \
             patch("ezexl3.repo.Process"):
            repo.run_measure_stage(
                model_dir="/tmp/model",
                bpws=["2"],
                devices=[0],
                write_logs=False,
                measure_args=[],
            )

        # file_size_gib called for the BPW with missing GiB
        mock_gib.assert_any_call("/tmp/model/2")
        # upsert_row called to fill the GiB gap
        mock_upsert.assert_any_call("/tmp/model/modelMeasured.db", weights="2", gib="4.2")

    def test_gib_gaps_not_filled_when_present(self):
        """No upsert when GiB values are already present."""
        full_rows = {
            "2": {"weights": "2", "KL Div": "0.1", "PPL": "11.0", "GiB": "4.2"},
            "bf16": {"weights": "bf16", "KL Div": "0.0", "PPL": "10.0", "GiB": "12.3"},
        }
        with patch("ezexl3.repo._read_db_rows", return_value=full_rows), \
             patch("ezexl3.repo.migrate_csv_to_db"), \
             patch("ezexl3.repo.file_size_gib") as mock_gib, \
             patch("ezexl3.repo.upsert_row") as mock_upsert, \
             patch("ezexl3.repo.Process"):
            repo.run_measure_stage(
                model_dir="/tmp/model",
                bpws=["2"],
                devices=[0],
                write_logs=False,
                measure_args=[],
            )

        # file_size_gib never called — no gaps to fill
        mock_gib.assert_not_called()


if __name__ == "__main__":
    unittest.main()
