import csv
import tempfile
import unittest
from unittest.mock import patch

from ezexl3 import cli
from ezexl3 import repo


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
            "2": {"weights": "2", "KL Div": "0.1", "PPL r-100": "11.0", "GiB": "4.2"},
            "bf16": {"weights": "bf16", "KL Div": "0.0", "PPL r-100": "10.0", "GiB": "12.3"},
        }
        with patch("ezexl3.repo._merge_csvs"), \
             patch("ezexl3.repo._read_csv_rows", return_value=full_rows), \
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

    def test_merge_csvs_preserves_existing_checkpoint_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = f"{tmp}/ModelMeasured.csv"
            shard_csv = f"{tmp}/ModelMeasured.gpu0.csv"
            fields = ["weights", "KL Div", "PPL r-100", "GiB"]

            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow({"weights": "bf16", "KL Div": "0.0", "PPL r-100": "10.0", "GiB": "12.3"})

            with open(shard_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow({"weights": "2", "KL Div": "0.1", "PPL r-100": "11.0", "GiB": "4.2"})

            repo._merge_csvs(out_csv, [shard_csv])

            with open(out_csv, "r", newline="") as f:
                rows = list(csv.DictReader(f))

        labels = [r["weights"] for r in rows]
        self.assertEqual(labels, ["bf16", "2"])

    def test_merge_csvs_combines_partial_rows(self):
        """KL-only and PPL-only rows for the same label merge into one complete row."""
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = f"{tmp}/ModelMeasured.csv"
            shard_a = f"{tmp}/ModelMeasured.gpu0.csv"
            shard_b = f"{tmp}/ModelMeasured.gpu1.csv"
            fields = ["weights", "KL Div", "PPL r-100", "GiB"]

            # No existing output CSV
            with open(shard_a, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow({"weights": "4", "KL Div": "0.05", "PPL r-100": "", "GiB": "6.0"})

            with open(shard_b, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow({"weights": "4", "KL Div": "", "PPL r-100": "9.8", "GiB": "6.0"})

            repo._merge_csvs(out_csv, [shard_a, shard_b])

            with open(out_csv, "r", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["weights"], "4")
        self.assertEqual(rows[0]["KL Div"], "0.05")
        self.assertEqual(rows[0]["PPL r-100"], "9.8")

    def test_merge_csvs_prefers_newer_shard_rows_for_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_csv = f"{tmp}/ModelMeasured.csv"
            shard_csv = f"{tmp}/ModelMeasured.gpu0.csv"
            fields = ["weights", "KL Div", "PPL r-100", "GiB"]

            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow({"weights": "2", "KL Div": "9.9", "PPL r-100": "99.0", "GiB": "4.0"})

            with open(shard_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerow({"weights": "2", "KL Div": "0.2", "PPL r-100": "12.0", "GiB": "4.1"})

            repo._merge_csvs(out_csv, [shard_csv])

            with open(out_csv, "r", newline="") as f:
                rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["weights"], "2")
        self.assertEqual(rows[0]["KL Div"], "0.2")


if __name__ == "__main__":
    unittest.main()
