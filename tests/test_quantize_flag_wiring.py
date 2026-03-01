import unittest
from unittest.mock import patch

from ezexl3 import cli
from ezexl3 import repo


class QuantizeFlagWiringTests(unittest.TestCase):
    def test_run_quant_stage_forwards_quantize_options(self):
        with patch("ezexl3.repo.quant_run", return_value=0) as mock_quant_run:
            rc = repo.run_quant_stage(
                model_dir="/tmp/model",
                bpws=["2", "4"],
                devices=[0, 1],
                device_ratios="1,1",
                quant_args=["--foo", "bar"],
                out_template="{model}/out-{bpw}",
                w_template="{model}/work-{bpw}",
                dry_run=True,
                continue_on_error=True,
                optimized_measure_layers=3,
            )

        self.assertEqual(rc, 0)
        mock_quant_run.assert_called_once_with(
            models=["/tmp/model"],
            bpws=["2", "4"],
            forwarded=["--foo", "bar", "-d", "0,1", "-dr", "1,1"],
            out_template="{model}/out-{bpw}",
            w_template="{model}/work-{bpw}",
            dry_run=True,
            continue_on_error=True,
        )

    def test_cli_quantize_passes_options_into_run_quant_stage(self):
        argv = [
            "quantize",
            "-m",
            "/tmp/model",
            "-b",
            "2",
            "4",
            "-d",
            "0,1",
            "-r",
            "1,1",
            "--out-template",
            "{model}/custom-out-{bpw}",
            "--w-template",
            "{model}/custom-work-{bpw}",
            "--dry",
            "--continue-on-error",
            "-l",
            "1",
            "--quant-args",
            "--",
            "--foo",
            "bar",
        ]

        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_stage:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        mock_stage.assert_called_once_with(
            model_dir="/tmp/model",
            bpws=["2", "4"],
            devices=[0, 1],
            device_ratios="1,1",
            quant_args=["--foo", "bar"],
            out_template="{model}/custom-out-{bpw}",
            w_template="{model}/custom-work-{bpw}",
            dry_run=True,
            continue_on_error=True,
            optimized_measure_layers=1,
        )


class QuantizeDecimalBpwTests(unittest.TestCase):
    """Tests for decimal bitrate quantization support in the quantize subcommand."""

    def test_cli_quantize_decimal_bpw_plans_and_optimizes(self):
        """Decimal BPW should quantize integer donors then run optimization."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.07", "-d", "0,1"]

        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        # Quant stage should receive integer donors [4, 5]
        self.assertEqual(mock_quant.call_args.kwargs["bpws"], ["4", "5"])
        # Optimization stage should receive the decimal BPW
        mock_opt.assert_called_once_with(
            model_dir="/tmp/model",
            optimized_bpws=["4.07"],
            devices=[0, 1],
            layers=2,
            write_logs=True,
        )

    def test_cli_quantize_mixed_bpws(self):
        """Mixed integer+decimal BPWs should separate correctly."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "2", "4.07", "-d", "0"]

        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        # Quant stage gets requested integer (2) + donors (4, 5)
        self.assertEqual(mock_quant.call_args.kwargs["bpws"], ["2", "4", "5"])
        mock_opt.assert_called_once()
        self.assertEqual(mock_opt.call_args.kwargs["optimized_bpws"], ["4.07"])

    def test_cli_quantize_integer_only_skips_optimization(self):
        """Integer-only BPWs should NOT trigger optimization stage."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "2", "3", "-d", "0"]

        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        self.assertEqual(mock_quant.call_args.kwargs["bpws"], ["2", "3"])
        mock_opt.assert_not_called()

    def test_cli_quantize_dry_run_skips_optimization(self):
        """--dry mode should skip the optimization stage."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.07", "-d", "0", "--dry"]

        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        mock_quant.assert_called_once()
        mock_opt.assert_not_called()

    def test_cli_quantize_custom_template_with_decimal_errors(self):
        """Custom --out-template with decimal BPWs should error."""
        argv = [
            "quantize", "-m", "/tmp/model", "-b", "4.07",
            "--out-template", "{model}/custom-{bpw}",
        ]

        with patch("ezexl3.repo.run_quant_stage", return_value=0), \
             patch("ezexl3.repo._run_optimized_opt_stage"):
            rc = cli.main(argv)

        self.assertEqual(rc, 1)

    def test_cli_quantize_layers_passed_to_optimization(self):
        """The -l/--layers flag should be passed to the optimization stage."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.07", "-d", "0", "-l", "3"]

        with patch("ezexl3.repo.run_quant_stage", return_value=0), \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        self.assertEqual(mock_opt.call_args.kwargs["layers"], 3)

    def test_cli_quantize_no_logs_passed_to_optimization(self):
        """The --no-logs flag should disable logs in the optimization stage."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.07", "-d", "0", "--no-logs"]

        with patch("ezexl3.repo.run_quant_stage", return_value=0), \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        self.assertEqual(mock_opt.call_args.kwargs["write_logs"], False)

    def test_cli_quantize_quant_failure_skips_optimization(self):
        """If quantization fails, optimization should be skipped."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.07", "-d", "0"]

        with patch("ezexl3.repo.run_quant_stage", return_value=1) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 1)
        mock_quant.assert_called_once()
        mock_opt.assert_not_called()


if __name__ == "__main__":
    unittest.main()
