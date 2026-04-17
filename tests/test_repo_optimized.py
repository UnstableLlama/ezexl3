import importlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from ezexl3 import repo


class RepoOptimizedModuleTests(unittest.TestCase):
    def test_module_build_optimized_jobs_matches_repo_wrapper(self):
        repo_optimized = importlib.import_module("ezexl3.repo_optimized")
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            for d in ["2", "3", "4", "5"]:
                (model_dir / d).mkdir()

            module_jobs = repo_optimized._build_optimized_jobs(str(model_dir), ["2.3", "2.7", "4.1"])
            repo_jobs = repo._build_optimized_jobs(str(model_dir), ["2.3", "2.7", "4.1"])

        self.assertEqual(module_jobs, repo_jobs)

    def test_module_compare_queue_prints_start_and_done(self):
        repo_optimized = importlib.import_module("ezexl3.repo_optimized")
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
                self.args = args

            def start(self):
                self.args[4].put({"event": "start", "device": 1, "job": jobs[0]})
                self.args[4].put({"event": "done", "device": 1, "job": jobs[0], "label": "3-4"})
                self.args[4].put(None)

            def join(self):
                return None

        buf = io.StringIO()
        with patch("ezexl3.repo_optimized.Process", DummyProcess), redirect_stdout(buf):
            repo_optimized._run_optimized_compare_queue(
                model_dir="/tmp/model",
                compare_jobs=jobs,
                devices=[1],
                layers=2,
                write_logs=False,
            )

        printed = buf.getvalue()
        self.assertIn("[GPU 1] START compare 3-4", printed)
        self.assertIn("[GPU 1] DONE compare 3-4", printed)


if __name__ == "__main__":
    unittest.main()
