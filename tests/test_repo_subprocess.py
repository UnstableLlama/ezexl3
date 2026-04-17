import importlib
import io
import sys
import tempfile
import unittest
from multiprocessing import Queue
from pathlib import Path

from ezexl3 import repo


class RepoSubprocessModuleTests(unittest.TestCase):
    def test_run_cmd_runs_successfully(self):
        repo_subprocess = importlib.import_module("ezexl3.repo_subprocess")
        repo_subprocess._run_cmd([sys.executable, "-c", "print('ok')"])

    def test_run_cmd_raises_on_failure(self):
        repo_subprocess = importlib.import_module("ezexl3.repo_subprocess")
        with self.assertRaises(RuntimeError):
            repo_subprocess._run_cmd([sys.executable, "-c", "import sys; sys.exit(3)"])

    def test_run_cmd_with_progress_captures_output(self):
        repo_subprocess = importlib.import_module("ezexl3.repo_subprocess")
        results = Queue()
        log_buf = io.StringIO()
        cmd = [sys.executable, "-c", "print('hello from subprocess')"]

        output = repo_subprocess._run_cmd_with_progress(cmd, device=0, results=results, log_f=log_buf)

        self.assertIn("hello from subprocess", output)
        self.assertIn("hello from subprocess", log_buf.getvalue())

    def test_run_measure_subprocess_emits_final_progress(self):
        repo_subprocess = importlib.import_module("ezexl3.repo_subprocess")
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "measure_like.py"
            script.write_text(
                "print('Processing 3 layers')\n"
                "print(' -- model.embed  time: 0.1')\n"
                "print(' -- model.layers.0  time: 0.2')\n"
                "print(' -- model.head  time: 0.3')\n"
                "print('Perplexity: 12.34')\n"
            )
            results = Queue()
            output = repo_subprocess._run_measure_subprocess(
                [sys.executable, str(script)],
                device=1,
                results=results,
                phase_label="4 PPL",
            )

        self.assertIn("Perplexity: 12.34", output)
        events = []
        while not results.empty():
            events.append(results.get())
        self.assertTrue(any(event["event"] == "progress" for event in events))
        self.assertTrue(any("100%" in event["text"] for event in events if event["event"] == "progress"))

    def test_repo_reexports_subprocess_helpers(self):
        repo_subprocess = importlib.import_module("ezexl3.repo_subprocess")
        self.assertIs(repo._run_cmd, repo_subprocess._run_cmd)
        self.assertIs(repo._run_cmd_with_progress, repo_subprocess._run_cmd_with_progress)
        self.assertIs(repo._run_measure_subprocess, repo_subprocess._run_measure_subprocess)
        self.assertIs(repo._run_catbench_subprocess, repo_subprocess._run_catbench_subprocess)


if __name__ == "__main__":
    unittest.main()
