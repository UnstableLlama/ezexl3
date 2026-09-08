import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Keep the module import GPU-free (same convention as the other CLI tests).
if "torch" not in sys.modules:
    _mock_torch = MagicMock()
    _mock_torch.cuda.is_available.return_value = False
    _mock_torch.cuda.device_count.return_value = 0
    sys.modules["torch"] = _mock_torch
if "exllamav3" not in sys.modules:
    sys.modules["exllamav3"] = MagicMock()

import tempfile

from ezexl3 import cli, repo_plan, repo_selfcal


def _make_quant_dir(model_dir: str, name: str) -> str:
    path = os.path.join(model_dir, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "config.json"), "w") as f:
        f.write("{}")
    return path


class SelfcalPlanTests(unittest.TestCase):
    def test_sc_bpws_leave_plain_quant_queue(self):
        plan = repo_plan._plan_repo_bpws(["2", "3", "3.14"], sc_bpws={"3.14", "2"})
        self.assertEqual(plan["requested_selfcal"], ["2", "3.14"])
        self.assertEqual(plan["quant_integer_queue"], ["3"])
        self.assertEqual(plan["measure_queue"], ["2", "3", "3.14"])

    def test_sc_and_opt_on_same_bpw_conflict(self):
        with self.assertRaises(ValueError):
            repo_plan._plan_repo_bpws(
                ["4.5"], opt_bpws={"4.5"}, sc_bpws={"4.5"}
            )

    def test_sc_on_opt_donor_neighbor_conflict(self):
        # 4.5 with -opt needs plain 4 and 5; painting 5 with -sc would build a
        # different quant at the same output path.
        with self.assertRaises(ValueError):
            repo_plan._plan_repo_bpws(
                ["4.5", "5"], opt_bpws={"4.5"}, sc_bpws={"5"}
            )

    def test_plan_without_sc_keeps_existing_shape(self):
        plan = repo_plan._plan_repo_bpws(["4", "4.07"], opt_bpws={"4.07"})
        self.assertEqual(plan["requested_selfcal"], [])
        self.assertEqual(plan["quant_integer_queue"], ["4", "5"])


class SelfcalDiscoveryTests(unittest.TestCase):
    def test_trace_donor_prefers_highest_quant_at_or_above_min(self):
        with tempfile.TemporaryDirectory() as model_dir:
            _make_quant_dir(model_dir, "2")
            five = _make_quant_dir(model_dir, "5")
            six = _make_quant_dir(model_dir, "6")
            os.makedirs(os.path.join(model_dir, "w-4"))  # ignored: no config
            self.assertEqual(repo_selfcal._find_trace_donor(model_dir), six)
            os.remove(os.path.join(six, "config.json"))
            self.assertEqual(repo_selfcal._find_trace_donor(model_dir), five)

    def test_trace_donor_none_when_only_low_bpw_quants(self):
        with tempfile.TemporaryDirectory() as model_dir:
            _make_quant_dir(model_dir, "2")
            _make_quant_dir(model_dir, "4.5")
            self.assertIsNone(repo_selfcal._find_trace_donor(model_dir))

    def test_probe_anchor_is_lowest_integer_quant(self):
        with tempfile.TemporaryDirectory() as model_dir:
            _make_quant_dir(model_dir, "2.5")  # fractional: not an anchor
            _make_quant_dir(model_dir, "4")
            three = _make_quant_dir(model_dir, "3")
            self.assertEqual(repo_selfcal._find_probe_anchor(model_dir), ("3", three))

    def test_probe_anchor_none_without_integer_quants(self):
        with tempfile.TemporaryDirectory() as model_dir:
            _make_quant_dir(model_dir, "2.5")
            self.assertIsNone(repo_selfcal._find_probe_anchor(model_dir))


class SelfcalStageTests(unittest.TestCase):
    def _run(self, model_dir, sc_bpws, head_bits=None, run_script_fn=None, devices=None):
        script_calls = []
        quant_calls = []

        def fake_run_script(cmd, env_extra=None, log_path=None):
            script_calls.append({"cmd": cmd, "env": env_extra, "log": log_path})
            if run_script_fn:
                run_script_fn(cmd)

        def fake_quant_one(mdir, bpw, forwarded, out_tmpl, w_tmpl):
            quant_calls.append({"bpw": bpw, "forwarded": forwarded})
            return True

        repo_selfcal.run_selfcal_stage(
            model_dir=model_dir,
            sc_bpws=sc_bpws,
            devices=[0, 1] if devices is None else devices,
            forwarded_for_bpw=lambda b: ["-d", "0,1"],
            head_bits=head_bits,
            write_logs=False,
            run_script_fn=fake_run_script,
            quant_one_fn=fake_quant_one,
            check_support_fn=lambda: None,
        )
        return script_calls, quant_calls

    def test_full_pipeline_from_scratch(self):
        with tempfile.TemporaryDirectory() as model_dir:
            _make_quant_dir(model_dir, "2")
            _make_quant_dir(model_dir, "6")
            script_calls, quant_calls = self._run(model_dir, ["3.14"], head_bits=4)

        scripts = [os.path.basename(c["cmd"][1]) for c in script_calls]
        self.assertEqual(
            scripts,
            ["sc_trace.py", "sc_rfn_probe.py", "sc_measure_parallel.py", "sc_optimize.py"],
        )

        trace = script_calls[0]
        self.assertIn(os.path.join(model_dir, "6"), trace["cmd"])  # donor quant
        self.assertEqual(trace["env"], {"CUDA_VISIBLE_DEVICES": "0,1"})

        probe = script_calls[1]
        self.assertIn(os.path.join(model_dir, "2"), probe["cmd"])  # lowest anchor

        measure = script_calls[2]
        self.assertIn("--shaped", measure["cmd"])
        self.assertIn("-rr", measure["cmd"])
        self.assertEqual(measure["cmd"][measure["cmd"].index("--devices") + 1], "0,1")

        optimize = script_calls[3]
        self.assertIn("-b", optimize["cmd"])
        self.assertIn("3.14", optimize["cmd"])
        self.assertIn("-hb", optimize["cmd"])
        self.assertIn("4", optimize["cmd"])

        self.assertEqual(len(quant_calls), 1)
        fwd = quant_calls[0]["forwarded"]
        self.assertIn("-rcp", fwd)
        self.assertIn("-cd", fwd)
        self.assertEqual(fwd[:2], ["-d", "0,1"])

    def test_trace_falls_back_to_unquantized_model(self):
        with tempfile.TemporaryDirectory() as model_dir:
            script_calls, _ = self._run(model_dir, ["3"])
        trace = script_calls[0]
        self.assertIn(os.path.abspath(model_dir), trace["cmd"])

    def test_probe_skipped_without_integer_quants(self):
        with tempfile.TemporaryDirectory() as model_dir:
            script_calls, _ = self._run(model_dir, ["3"])
        scripts = [os.path.basename(c["cmd"][1]) for c in script_calls]
        self.assertNotIn("sc_rfn_probe.py", scripts)
        measure = next(c for c in script_calls if "sc_measure" in c["cmd"][1])
        self.assertNotIn("-rr", measure["cmd"])
        self.assertEqual(measure["cmd"][measure["cmd"].index("--load-mode") + 1], "auto")

    def test_completed_stages_are_skipped_on_resume(self):
        with tempfile.TemporaryDirectory() as model_dir:
            paths = repo_selfcal._selfcal_paths(model_dir)
            os.makedirs(paths["dir"])
            for key in ("trace_json", "trace_st", "rfn", "attrib", "attrib_done"):
                with open(paths[key], "w") as f:
                    f.write("stub")
            with open(repo_selfcal._recipe_path(model_dir, "3"), "w") as f:
                f.write("stub")
            _make_quant_dir(model_dir, "3")  # conversion output already exists

            script_calls, quant_calls = self._run(model_dir, ["3"])

        self.assertEqual(script_calls, [])
        self.assertEqual(quant_calls, [])

    def test_arbitrary_device_list_uses_logical_indices(self):
        with tempfile.TemporaryDirectory() as model_dir:
            calls, _ = self._run(model_dir, ["3"], devices=[2, 4, 6, 7])
        measure = next(c for c in calls if "sc_measure" in c["cmd"][1])
        self.assertEqual(measure["env"]["CUDA_VISIBLE_DEVICES"], "2,4,6,7")
        self.assertEqual(measure["cmd"][measure["cmd"].index("--devices") + 1], "0,1,2,3")

    def test_single_gpu_uses_original_worker_unless_recovering_shards(self):
        with tempfile.TemporaryDirectory() as model_dir:
            calls, _ = self._run(model_dir, ["3"], devices=[4])
            measure = next(c for c in calls if "sc_measure" in c["cmd"][1])
            self.assertEqual(os.path.basename(measure["cmd"][1]), "sc_measure.py")
            paths = repo_selfcal._selfcal_paths(model_dir)
            os.remove(paths["attrib_done"])
            os.makedirs(paths["attrib"] + ".workers")
            calls, _ = self._run(model_dir, ["3"], devices=[4])
            measure = next(c for c in calls if "sc_measure" in c["cmd"][1])
            self.assertEqual(os.path.basename(measure["cmd"][1]), "sc_measure_parallel.py")
            self.assertEqual(measure["cmd"][measure["cmd"].index("--devices") + 1], "0")

    def test_worker_failure_does_not_mark_stage_complete(self):
        def fail(cmd):
            if "sc_measure" in cmd[1]:
                raise RuntimeError("worker failed")

        with tempfile.TemporaryDirectory() as model_dir:
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                self._run(model_dir, ["3"], run_script_fn=fail)
            paths = repo_selfcal._selfcal_paths(model_dir)
            self.assertFalse(os.path.exists(paths["attrib_done"]))

    def test_cancelled_script_interrupts_coordinator_before_terminating(self):
        import signal
        import subprocess
        proc = MagicMock()
        proc.stdout.read1.side_effect = KeyboardInterrupt
        proc.poll.return_value = None
        with patch.object(repo_selfcal.subprocess, "Popen", return_value=proc):
            with self.assertRaises(KeyboardInterrupt):
                repo_selfcal._run_script(["python", "sc_measure_parallel.py"])
        # SIGINT lets the worker's finally blocks release torch's build lock.
        proc.send_signal.assert_called_once_with(signal.SIGINT)
        proc.wait.assert_called_once_with(timeout=15)
        proc.terminate.assert_not_called()

        proc.reset_mock()
        proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 15), subprocess.TimeoutExpired("cmd", 5), 0]
        with patch.object(repo_selfcal.subprocess, "Popen", return_value=proc):
            with self.assertRaises(KeyboardInterrupt):
                repo_selfcal._run_script(["python", "sc_measure_parallel.py"])
        proc.send_signal.assert_called_once_with(signal.SIGINT)
        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_unsupported_exllamav3_raises(self):
        with tempfile.TemporaryDirectory() as model_dir:
            with self.assertRaises(RuntimeError):
                repo_selfcal.run_selfcal_stage(
                    model_dir=model_dir,
                    sc_bpws=["3"],
                    devices=[0],
                    forwarded_for_bpw=lambda b: [],
                    check_support_fn=lambda: "too old",
                )


class SelfcalCliWiringTests(unittest.TestCase):
    def test_repo_passes_sc_and_head_bits(self):
        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run:
            rc = cli.main([
                "repo", "-m", "/tmp/model", "-b", "3,4.5",
                "-sc", "4.5", "-hb", "5", "-vb", "16", "-np",
            ])
        self.assertEqual(rc, 0)
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["sc_bpws"], {"4.5"})
        self.assertEqual(kwargs["head_bits"], 5)
        self.assertEqual(kwargs["quant_args"], ["-hb", "5", "-vb", "16"])

    def test_bare_sc_applies_to_all_bpws(self):
        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run:
            cli.main(["repo", "-m", "/tmp/model", "-b", "2,3", "-sc", "-np"])
        self.assertEqual(mock_run.call_args.kwargs["sc_bpws"], {"2", "3"})

    def test_sc_opt_conflict_rejected(self):
        with self.assertRaises(SystemExit):
            cli.main([
                "repo", "-m", "/tmp/model", "-b", "4.5",
                "-sc", "4.5", "-opt", "4.5", "-np",
            ])

    def test_head_bits_out_of_range_rejected(self):
        with self.assertRaises(SystemExit):
            cli.main(["repo", "-m", "/tmp/model", "-b", "4", "-hb", "9", "-np"])

    def test_vision_bits_allows_16(self):
        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run:
            cli.main(["repo", "-m", "/tmp/model", "-b", "4", "-vb", "16", "-np"])
        self.assertEqual(mock_run.call_args.kwargs["quant_args"], ["-vb", "16"])

    def test_vision_bits_out_of_range_rejected(self):
        with self.assertRaises(SystemExit):
            cli.main(["repo", "-m", "/tmp/model", "-b", "4", "-vb", "12", "-np"])


if __name__ == "__main__":
    unittest.main()
