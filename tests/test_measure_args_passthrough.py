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


if __name__ == "__main__":
    unittest.main()
