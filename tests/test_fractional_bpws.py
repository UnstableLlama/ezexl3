import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ezexl3 import repo


class FractionalBpwPlanningTests(unittest.TestCase):
    def test_plan_repo_bpws_adds_integer_neighbors(self):
        plan = repo._plan_repo_bpws(["4", "4.07", "6.25"])

        self.assertEqual(plan["requested_integers"], ["4"])
        self.assertEqual(plan["requested_fractionals"], ["4.07", "6.25"])
        self.assertEqual(plan["quant_integer_queue"], ["4", "5", "6", "7"])
        self.assertEqual(plan["measure_queue"], ["4", "5", "6", "7", "4.07", "6.25"])

    def test_plan_repo_bpws_normalizes_integer_like_tokens(self):
        plan = repo._plan_repo_bpws(["4.0", "5.00", "4.10"])

        self.assertEqual(plan["requested_integers"], ["4", "5"])
        self.assertEqual(plan["requested_fractionals"], ["4.1"])


class FractionalStageTests(unittest.TestCase):
    def test_build_fractional_jobs_dedupes_shared_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            for d in ["2", "3", "4", "5"]:
                (model_dir / d).mkdir()

            compare_jobs, optimize_jobs = repo._build_fractional_jobs(str(model_dir), ["2.3", "2.7", "4.1"])

            self.assertEqual(len(compare_jobs), 2)
            labels = {(j["low"], j["high"]): j for j in compare_jobs}
            self.assertEqual(sorted(labels[("2", "3")]["targets"]), ["2.3", "2.7"])
            self.assertEqual(labels[("4", "5")]["targets"], ["4.1"])
            self.assertEqual(len(optimize_jobs), 3)

    def test_fractional_stage_skips_existing_measurement_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "4").mkdir()
            (model_dir / "5").mkdir()
            measurements = model_dir / "measurements"
            measurements.mkdir()
            (measurements / "4-5_measurement.json").write_text("{}")

            with patch(
                "ezexl3.repo._resolve_exllamav3_util_scripts",
                return_value=("/opt/exl3/util/measure.py", "/opt/exl3/util/optimize.py"),
            ), patch("ezexl3.repo._run_fractional_compare_queue") as mock_queue, patch(
                "ezexl3.repo._run_cmd"
            ) as mock_run:
                repo._run_fractional_opt_stage(str(model_dir), ["4.07"], devices=[0, 1], write_logs=False)

            mock_queue.assert_called_once()
            queued_jobs = mock_queue.call_args.kwargs["compare_jobs"]
            self.assertEqual(queued_jobs, [])
            self.assertEqual(mock_run.call_count, 1)
            cmd = mock_run.call_args.args[0]
            self.assertEqual(cmd[0:2], [repo.sys.executable, "/opt/exl3/util/optimize.py"])
            self.assertIn("4.07", cmd)

    def test_compare_queue_prints_start_and_done(self):
        jobs = [
            {
                "low": "3",
                "high": "4",
                "targets": ["3.5"],
                "measure_json": "/tmp/m.json",
                "low_dir": "/tmp/3",
                "high_dir": "/tmp/4",
            }
        ]

        class DummyProcess:
            def __init__(self, target, args):
                self.target = target
                self.args = args

            def start(self):
                # args=(measure_script, model_dir, device, primary_device, tasks, results, log_path)
                self.args[5].put({"event": "start", "device": 1, "job": jobs[0]})
                self.args[5].put({"event": "done", "device": 1, "job": jobs[0], "label": "3-4"})
                self.args[5].put(None)

            def join(self):
                return None

        with patch("ezexl3.repo.Process", DummyProcess), patch("builtins.print") as mock_print:
            repo._run_fractional_compare_queue(
                model_dir="/tmp/model",
                compare_jobs=jobs,
                devices=[1],
                measure_script="/opt/exl3/util/measure.py",
                write_logs=False,
            )

        printed = "\n".join(" ".join(str(x) for x in call.args) for call in mock_print.call_args_list)
        self.assertIn("[GPU 1] START compare 3-4", printed)
        self.assertIn("[GPU 1] DONE compare 3-4", printed)

    def test_run_repo_uses_planned_queues(self):
        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, patch(
            "ezexl3.repo._run_fractional_opt_stage"
        ) as mock_frac, patch("ezexl3.repo.run_measure_stage", return_value=0) as mock_measure:
            rc = repo.run_repo(
                model_dir="/tmp/model",
                bpws=["4.07"],
                devices=[0, 1],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_readme=False,
            )

        self.assertEqual(rc, 0)
        self.assertEqual(mock_quant.call_args.kwargs["bpws"], ["4", "5"])
        self.assertEqual(mock_measure.call_args.kwargs["bpws"], ["4", "5", "4.07"])
        mock_frac.assert_called_once_with(
            model_dir="/tmp/model", fractional_bpws=["4.07"], devices=[0, 1], write_logs=True
        )


if __name__ == "__main__":
    unittest.main()
