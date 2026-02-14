import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ezexl3 import cli
from ezexl3.readme import run_readme


class NoMeasurementFlagTests(unittest.TestCase):
    def test_repo_no_measurement_wiring(self):
        argv = [
            "repo",
            "-m",
            "/tmp/model",
            "-b",
            "2",
            "-nm",
        ]

        with patch("ezexl3.repo.run_repo", return_value=0) as mock_run_repo:
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        kwargs = mock_run_repo.call_args.kwargs
        self.assertFalse(kwargs["do_measure"])
        self.assertFalse(kwargs["include_graph"])
        self.assertFalse(kwargs["include_measurements"])

    def test_readme_no_measurement_removes_columns_and_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp) / "Author-Model"
            model_dir.mkdir()
            (model_dir / "2").mkdir()

            run_readme(
                str(model_dir),
                template_name="basic",
                interactive=False,
                include_graph=False,
                include_measurements=False,
                bpws_hint=["2"],
            )

            readme = (model_dir / "README.md").read_text()
            self.assertNotIn("<th>KL DIV</th>", readme)
            self.assertNotIn("<th>PPL</th>", readme)
            self.assertNotIn('class="repo-graph"', readme)
            self.assertIn('<th>REVISION</th>', readme)
            self.assertIn('<th>GiB</th>', readme)


if __name__ == "__main__":
    unittest.main()
