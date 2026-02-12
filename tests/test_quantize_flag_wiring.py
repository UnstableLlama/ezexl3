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
        )


if __name__ == "__main__":
    unittest.main()
