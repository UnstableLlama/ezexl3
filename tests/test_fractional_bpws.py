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
    def test_fractional_stage_skips_existing_measurement_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "4").mkdir()
            (model_dir / "5").mkdir()
            measurements = model_dir / "measurements"
            measurements.mkdir()
            (measurements / "4-5_measurement.json").write_text("{}")

            with patch("ezexl3.repo._resolve_exllamav3_util_scripts", return_value=("/opt/exl3/util/measure.py", "/opt/exl3/util/optimize.py")), patch(
                "ezexl3.repo._run_cmd"
            ) as mock_run:
                repo._run_fractional_opt_stage(str(model_dir), ["4.07"], device=0)

            self.assertEqual(mock_run.call_count, 1)
            cmd = mock_run.call_args.args[0]
            self.assertEqual(cmd[0:2], [repo.sys.executable, "/opt/exl3/util/optimize.py"])
            self.assertIn("4.07", cmd)

    def test_run_repo_uses_planned_queues(self):
        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, patch(
            "ezexl3.repo._run_fractional_opt_stage"
        ) as mock_frac, patch("ezexl3.repo.run_measure_stage", return_value=0) as mock_measure:
            rc = repo.run_repo(
                model_dir="/tmp/model",
                bpws=["4.07"],
                devices=[0],
                device_ratios=None,
                quant_args=[],
                measure_args=[],
                do_readme=False,
            )

        self.assertEqual(rc, 0)
        self.assertEqual(mock_quant.call_args.kwargs["bpws"], ["4", "5"])
        self.assertEqual(mock_measure.call_args.kwargs["bpws"], ["4", "5", "4.07"])
        mock_frac.assert_called_once_with(model_dir="/tmp/model", fractional_bpws=["4.07"], device=0)


if __name__ == "__main__":
    unittest.main()
