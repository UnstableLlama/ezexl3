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
            hq_bpws=set(),
            hb8_bpws=set(),
        )


class QuantizeDecimalBpwTests(unittest.TestCase):
    """Tests for decimal bitrate quantization support in the quantize subcommand."""

    def test_cli_quantize_decimal_bpw_without_opt_quants_directly(self):
        """Decimal BPW without -opt should quantize directly (no optimization)."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.07", "-d", "0,1"]

        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        # Without -opt, 4.07 goes straight to quant queue
        self.assertEqual(mock_quant.call_args.kwargs["bpws"], ["4.07"])
        mock_opt.assert_not_called()

    def test_cli_quantize_decimal_bpw_with_opt_plans_and_optimizes(self):
        """Decimal BPW with -opt should quantize integer donors then run optimization."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.07", "-d", "0,1", "-opt", "4.07"]

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
        """Mixed integer+decimal BPWs without -opt: fractionals quant directly."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "2", "4.07", "-d", "0"]

        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        # Without -opt, 4.07 goes into quant queue alongside 2
        self.assertEqual(mock_quant.call_args.kwargs["bpws"], ["2", "4.07"])
        mock_opt.assert_not_called()

    def test_cli_quantize_mixed_bpws_with_opt(self):
        """Mixed integer+decimal BPWs with -opt should separate correctly."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "2", "4.07", "-d", "0", "-opt", "4.07"]

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

    def test_cli_quantize_custom_template_with_opt_decimal_errors(self):
        """Custom --out-template with -opt decimal BPWs should error."""
        argv = [
            "quantize", "-m", "/tmp/model", "-b", "4.07",
            "--out-template", "{model}/custom-{bpw}", "-opt", "4.07",
        ]

        with patch("ezexl3.repo.run_quant_stage", return_value=0), \
             patch("ezexl3.repo._run_optimized_opt_stage"):
            rc = cli.main(argv)

        self.assertEqual(rc, 1)

    def test_cli_quantize_custom_template_with_decimal_without_opt_ok(self):
        """Custom --out-template with decimal BPWs without -opt should work fine."""
        argv = [
            "quantize", "-m", "/tmp/model", "-b", "4.07",
            "--out-template", "{model}/custom-{bpw}",
        ]

        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        mock_opt.assert_not_called()

    def test_cli_quantize_layers_passed_to_optimization(self):
        """The -l/--layers flag should be passed to the optimization stage."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.07", "-d", "0", "-l", "3", "-opt", "4.07"]

        with patch("ezexl3.repo.run_quant_stage", return_value=0), \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        self.assertEqual(mock_opt.call_args.kwargs["layers"], 3)

    def test_cli_quantize_no_logs_passed_to_optimization(self):
        """The --no-logs flag should disable logs in the optimization stage."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.07", "-d", "0", "--no-logs", "-opt", "4.07"]

        with patch("ezexl3.repo.run_quant_stage", return_value=0), \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        self.assertEqual(mock_opt.call_args.kwargs["write_logs"], False)

    def test_cli_quantize_quant_failure_skips_optimization(self):
        """If quantization fails, optimization should be skipped."""
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.07", "-d", "0", "-opt", "4.07"]

        with patch("ezexl3.repo.run_quant_stage", return_value=1) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage") as mock_opt:
            rc = cli.main(argv)

        self.assertEqual(rc, 1)
        mock_quant.assert_called_once()
        mock_opt.assert_not_called()


class BpwForwardingNormalizationTests(unittest.TestCase):
    """_build_quant_forwarded_for_bpw must normalize both sides of the
    hq/hb8 lookup so flag-painted BPWs match the planner-normalized
    quant queue, regardless of trailing zeros or decimal form."""

    def _has(self, args, *needles):
        # Check that the entire needles sequence appears contiguously in args
        for i in range(len(args) - len(needles) + 1):
            if list(args[i:i + len(needles)]) == list(needles):
                return True
        return False

    def test_hq_matches_when_user_typed_trailing_zero(self):
        # User typed "4.0" in the UI; planner normalizes to "4"
        forwarded = repo._build_quant_forwarded_for_bpw(
            quant_args=[], devices=[0], device_ratios=None,
            bpw="4", hq_bpws={"4.0"}, hb8_bpws=None,
        )
        self.assertIn("-hq", forwarded)

    def test_hb8_matches_when_user_typed_trailing_zero(self):
        forwarded = repo._build_quant_forwarded_for_bpw(
            quant_args=[], devices=[0], device_ratios=None,
            bpw="6", hq_bpws=None, hb8_bpws={"6.0"},
        )
        self.assertTrue(self._has(forwarded, "-hb", "8"))

    def test_fractional_normalization(self):
        # User typed "5.50"; planner normalizes to "5.5"
        forwarded = repo._build_quant_forwarded_for_bpw(
            quant_args=[], devices=[0], device_ratios=None,
            bpw="5.5", hq_bpws={"5.50"}, hb8_bpws=None,
        )
        self.assertIn("-hq", forwarded)

    def test_hq_not_added_when_bpw_not_painted(self):
        forwarded = repo._build_quant_forwarded_for_bpw(
            quant_args=[], devices=[0], device_ratios=None,
            bpw="3", hq_bpws={"4", "5"}, hb8_bpws=None,
        )
        self.assertNotIn("-hq", forwarded)


class OptPaintPropagationTests(unittest.TestCase):
    """A fractional BPW painted with -opt is built by quantizing its
    integer neighbors and combining them. Any -hq / -hb8 painted on
    that fractional should propagate to the donor integers so they
    actually receive the flag at convert time."""

    def test_hq_propagates_from_opt_fractional_to_neighbors(self):
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.5", "-d", "0",
                "-opt", "4.5", "-hq", "4.5"]
        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage"):
            cli.main(argv)
        kwargs = mock_quant.call_args.kwargs
        self.assertIn("4", kwargs["hq_bpws"])
        self.assertIn("5", kwargs["hq_bpws"])

    def test_hb8_propagates_from_opt_fractional_to_neighbors(self):
        argv = ["quantize", "-m", "/tmp/model", "-b", "5.5", "-d", "0",
                "-opt", "5.5", "-hb8", "5.5"]
        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage"):
            cli.main(argv)
        kwargs = mock_quant.call_args.kwargs
        self.assertIn("5", kwargs["hb8_bpws"])
        self.assertIn("6", kwargs["hb8_bpws"])

    def test_no_propagation_when_fractional_not_in_opt(self):
        # 4.5 is fractional but NOT painted with -opt → standard fractional,
        # quantized directly. No propagation should happen.
        argv = ["quantize", "-m", "/tmp/model", "-b", "4.5", "-d", "0",
                "-hq", "4.5"]
        with patch("ezexl3.repo.run_quant_stage", return_value=0) as mock_quant, \
             patch("ezexl3.repo._run_optimized_opt_stage"):
            cli.main(argv)
        kwargs = mock_quant.call_args.kwargs
        # 4.5 should be in hq_bpws but its integer neighbors should NOT
        self.assertIn("4.5", kwargs["hq_bpws"])
        self.assertNotIn("4", kwargs["hq_bpws"])
        self.assertNotIn("5", kwargs["hq_bpws"])


if __name__ == "__main__":
    unittest.main()
