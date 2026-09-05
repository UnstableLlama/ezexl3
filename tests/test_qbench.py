"""Tests for the qbench quant-comparison stage (ezexl3 qbench).

Covers CLI flag wiring into run_qbench, project YAML generation, the
reuse-vs-regen behavior of the project file, the support check, and the
WebUI wiring (server allowlist, commands.js schema, nav button).
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ezexl3 import cli
from ezexl3 import qbench

REPO_ROOT = Path(__file__).resolve().parent.parent


def _make_model_dir(tmp, bpws=("3.0", "4.0")):
    model_dir = os.path.join(tmp, "TestModel")
    os.makedirs(model_dir)
    for b in bpws:
        os.makedirs(os.path.join(model_dir, b))
    return model_dir


class QbenchCliParserTests(unittest.TestCase):
    def test_defaults(self):
        parser = cli.build_parser()
        args = parser.parse_args(["qbench", "-m", "/tmp/model"])
        self.assertEqual(args.cmd, "qbench")
        self.assertIsNone(args.bpws)
        self.assertEqual(args.device, 0)
        self.assertEqual(args.rows, 10)
        self.assertEqual(args.length, 2048)
        self.assertEqual(args.dataset, "wiki2")
        self.assertEqual(args.template, "none")
        self.assertEqual(args.ref_engine, "exllamav3")
        self.assertFalse(args.no_noise_floor)
        self.assertFalse(args.regen)

    def test_cli_dispatch_calls_run_qbench(self):
        with patch("ezexl3.qbench.run_qbench", return_value=0) as mock_run:
            rc = cli.main([
                "qbench", "-m", "/tmp/model", "-b", "3,4", "-d", "1",
                "--rows", "20", "--template", "chat", "--regen",
            ])
        self.assertEqual(rc, 0)
        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["model_dir"], "/tmp/model")
        self.assertEqual(kwargs["bpws"], ["3", "4"])
        self.assertEqual(kwargs["device"], 1)
        self.assertEqual(kwargs["rows"], 20)
        self.assertEqual(kwargs["template"], "chat")
        self.assertTrue(kwargs["regen"])
        self.assertTrue(kwargs["noise_floor"])

    def test_run_qbench_error_returns_nonzero(self):
        with patch("ezexl3.qbench.run_qbench", side_effect=RuntimeError("boom")):
            rc = cli.main(["qbench", "-m", "/tmp/model"])
        self.assertEqual(rc, 1)


class QbenchProjectTests(unittest.TestCase):
    def test_build_project_structure(self):
        proj = qbench.build_project("/m/Model", ["3.0", "4.0"], rows=12, length=1024)
        self.assertEqual(proj["title"], "Model")
        self.assertEqual(proj["test_data"],
                         {"source": "wiki2", "rows": 12, "length": 1024, "stride": 1024})
        self.assertIs(proj["tokenizer"]["template"], False)
        models = proj["models"]
        self.assertEqual(models[0]["group"], "reference")
        self.assertEqual(models[0]["engine"], "exllamav3")
        self.assertEqual(models[0]["source"], os.path.abspath("/m/Model"))
        self.assertEqual([m["label"] for m in models[1:]], ["3.0 bpw", "4.0 bpw"])
        self.assertTrue(all(m["engine"] == "exllamav3" for m in models[1:]))
        self.assertEqual(proj["logit_cache"]["dir"], "logit_cache")
        self.assertIn("plot_kld_spread", proj["output"])

    def test_template_mapping(self):
        for opt, expect in (("none", False), ("chat", True), ("assistant", "assistant")):
            proj = qbench.build_project("/m/Model", ["4.0"], template=opt)
            self.assertEqual(proj["tokenizer"]["template"], expect)

    def test_transformers_reference_gets_streaming(self):
        proj = qbench.build_project("/m/Model", ["4.0"], ref_engine="transformers")
        self.assertEqual(proj["models"][0]["options"], {"streaming": True})

    def test_trace_replaces_test_data(self):
        proj = qbench.build_project("/m/Model", ["4.0"], trace="/tmp/trace.json")
        self.assertEqual(proj["test_trace"], os.path.abspath("/tmp/trace.json"))
        self.assertNotIn("test_data", proj)
        self.assertNotIn("tokenizer", proj)

    def test_missing_quants(self):
        proj = qbench.build_project("/m/Model", ["3.0"])
        self.assertEqual(qbench._missing_quants(proj, "/m/Model", ["3.0", "4.0"]),
                         ["4.0"])


class QbenchRunnerTests(unittest.TestCase):
    def _run(self, model_dir, **kwargs):
        with patch("ezexl3.qbench.run_cmd_capture", return_value="") as mock_cmd, \
             patch("ezexl3.qbench.check_qbench_support", return_value=None):
            rc = qbench.run_qbench(model_dir, **kwargs)
        return rc, mock_cmd

    def test_writes_project_and_invokes_vendored_script(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _make_model_dir(tmp)
            rc, mock_cmd = self._run(model_dir, bpws=["3.0", "4.0"], device=1)
            self.assertEqual(rc, 0)
            project_path = qbench.default_project_path(model_dir)
            self.assertTrue(os.path.isfile(project_path))
            cmd = mock_cmd.call_args[0][0]
            self.assertTrue(cmd[1].endswith(os.path.join("vendor", "eval", "qbench.py")))
            self.assertEqual(cmd[2], project_path)
            self.assertEqual(cmd[cmd.index("-d") + 1], "1")

    def test_auto_detects_bpws(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _make_model_dir(tmp, bpws=("2.5", "5.0"))
            os.makedirs(os.path.join(model_dir, "w-2.5"))  # workdir: not a quant
            rc, _ = self._run(model_dir)
            self.assertEqual(rc, 0)
            import yaml
            with open(qbench.default_project_path(model_dir)) as f:
                proj = yaml.safe_load(f)
            labels = [m["label"] for m in proj["models"][1:]]
            self.assertEqual(labels, ["2.5 bpw", "5.0 bpw"])

    def test_reuses_existing_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _make_model_dir(tmp)
            self._run(model_dir, bpws=["3.0"])
            project_path = qbench.default_project_path(model_dir)
            # Hand-edit survives a rerun without --regen
            with open(project_path, "a") as f:
                f.write("# hand edit\n")
            before = Path(project_path).read_text()
            rc, _ = self._run(model_dir, bpws=["3.0"])
            self.assertEqual(rc, 0)
            self.assertEqual(Path(project_path).read_text(), before)

    def test_regen_overwrites_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _make_model_dir(tmp)
            self._run(model_dir, bpws=["3.0"])
            project_path = qbench.default_project_path(model_dir)
            with open(project_path, "a") as f:
                f.write("# hand edit\n")
            rc, _ = self._run(model_dir, bpws=["3.0"], regen=True)
            self.assertEqual(rc, 0)
            self.assertNotIn("# hand edit", Path(project_path).read_text())

    def test_missing_quant_dir_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _make_model_dir(tmp, bpws=("3.0",))
            rc, mock_cmd = self._run(model_dir, bpws=["3.0", "9.9"])
            self.assertEqual(rc, 1)
            mock_cmd.assert_not_called()

    def test_no_quants_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _make_model_dir(tmp, bpws=())
            rc, mock_cmd = self._run(model_dir)
            self.assertEqual(rc, 1)
            mock_cmd.assert_not_called()

    def test_support_check_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = _make_model_dir(tmp)
            with patch("ezexl3.qbench.run_cmd_capture") as mock_cmd, \
                 patch("ezexl3.qbench.check_qbench_support", return_value="nope"):
                rc = qbench.run_qbench(model_dir, bpws=["3.0"])
            self.assertEqual(rc, 1)
            mock_cmd.assert_not_called()


class QbenchVendorTests(unittest.TestCase):
    def test_vendored_files_present_and_in_manifest(self):
        vendor_dir = REPO_ROOT / "ezexl3" / "vendor"
        manifest = json.loads((vendor_dir / "VENDOR_MANIFEST.json").read_text())
        for rel in (
            "eval/qbench.py",
            "eval/qbench/__init__.py",
            "eval/qbench/data.py",
            "eval/qbench/engines.py",
            "eval/qbench/measure.py",
            "eval/qbench/plot.py",
        ):
            self.assertTrue((vendor_dir / rel).is_file(), rel)
            self.assertIn(rel, manifest)


class QbenchUiWiringTests(unittest.TestCase):
    def test_server_allowlists_qbench(self):
        src = (REPO_ROOT / "ezexl3" / "ui" / "server.py").read_text()
        self.assertRegex(src, r'valid_commands = \{[^}]*"qbench"')

    def test_commands_js_folds_qbench_into_the_evals_form(self):
        src = (REPO_ROOT / "ezexl3" / "ui" / "static" / "js" / "commands.js").read_text()
        # No standalone qbench command any more — its knobs live in the
        # measure ("Evals") form's collapsed qbench group.
        self.assertNotIn("qbench: {\n    label:", src)
        self.assertIn('label: "Evals"', src)
        self.assertIn('subtitle: "Measure"', src)
        for flag in ("--rows", "--length", "--dataset", "--trace",
                     "--ref-engine", "--cache-gb", "--no-noise-floor", "--regen"):
            self.assertIn(f'flag: "{flag}"', src)
            line = next(ln for ln in src.splitlines() if f'flag: "{flag}"' in ln)
            self.assertIn('group: "qbench"', line, f"{flag} is not in the qbench group")

    def test_index_html_drops_the_qbench_nav_button(self):
        src = (REPO_ROOT / "ezexl3" / "ui" / "static" / "index.html").read_text()
        self.assertNotIn('data-cmd="qbench"', src)
        self.assertIn('data-cmd="measure"', src)

    def test_results_tab_offers_kl_ppl_alongside_the_other_evals(self):
        src = (REPO_ROOT / "ezexl3" / "ui" / "static" / "index.html").read_text()
        self.assertIn('data-tab="results"', src)
        for kind in ("klppl", "perf", "catbench"):
            self.assertIn(f'<option value="{kind}">', src)


class MeasureQbenchTuningTests(unittest.TestCase):
    """`measure` forwards only the qbench knobs the user actually typed."""

    UI_FLAGS = ["--rows", "50", "--length", "4096", "--dataset", "openwebtext",
                "--template", "chat", "--trace", "/t.json",
                "--ref-engine", "transformers", "--cache-gb", "20",
                "--no-noise-floor", "--regen"]

    def test_untouched_flags_are_not_forwarded(self):
        parser = cli.build_parser()
        args = parser.parse_args(["measure", "-m", "/m", "-b", "4"])
        self.assertEqual(cli._collect_qbench_opts(args), {})

    def test_typed_flags_map_onto_run_qbench_kwargs(self):
        parser = cli.build_parser()
        args = parser.parse_args(["measure", "-m", "/m", "-b", "4"] + self.UI_FLAGS)
        self.assertEqual(cli._collect_qbench_opts(args), {
            "rows": 50, "length": 4096, "dataset": "openwebtext",
            "template": "chat", "trace": "/t.json",
            "ref_engine": "transformers", "cache_gb": 20.0,
            "regen": True, "noise_floor": False,
        })

    def test_collected_opts_are_all_real_run_qbench_kwargs(self):
        import inspect
        parser = cli.build_parser()
        args = parser.parse_args(["measure", "-m", "/m", "-b", "4"] + self.UI_FLAGS)
        accepted = set(inspect.signature(qbench.run_qbench).parameters)
        self.assertEqual(set(cli._collect_qbench_opts(args)) - accepted, set())

    def test_measure_dispatch_passes_opts_to_the_stage(self):
        with patch("ezexl3.repo.run_measure_stage", return_value=0) as mock_stage:
            rc = cli.main(["measure", "-m", "/m", "-b", "4", "--rows", "50"])
        self.assertEqual(rc, 0)
        self.assertEqual(mock_stage.call_args.kwargs["qbench_opts"], {"rows": 50})

    def test_measure_dispatch_passes_none_when_untouched(self):
        with patch("ezexl3.repo.run_measure_stage", return_value=0) as mock_stage:
            rc = cli.main(["measure", "-m", "/m", "-b", "4"])
        self.assertEqual(rc, 0)
        self.assertIsNone(mock_stage.call_args.kwargs["qbench_opts"])


if __name__ == "__main__":
    unittest.main()
