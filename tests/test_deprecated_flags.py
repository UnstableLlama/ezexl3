import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from ezexl3 import cli


class DeprecatedFlagTests(unittest.TestCase):
    def test_repo_warns_for_deprecated_or_unused_flags(self):
        argv = [
            "repo",
            "-m",
            "/tmp/model",
            "-b",
            "2",
            "--schedule",
            "static",
            "--no-meta",
            "--exllamav3-root",
            "/tmp/exl3",
            "--no-readme",
        ]

        buf = io.StringIO()
        with patch("ezexl3.repo.run_repo", return_value=0), redirect_stdout(buf):
            rc = cli.main(argv)

        out = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("--exllamav3-root is deprecated and ignored", out)
        self.assertIn("--schedule is currently ignored", out)
        self.assertIn("--no-meta is currently ignored", out)

    def test_measure_warns_for_deprecated_exllamav3_root(self):
        argv = [
            "measure",
            "-m",
            "/tmp/model",
            "-b",
            "2",
            "--exllamav3-root",
            "/tmp/exl3",
        ]

        buf = io.StringIO()
        with patch("ezexl3.repo.run_measure_stage", return_value=0), redirect_stdout(buf):
            rc = cli.main(argv)

        self.assertEqual(rc, 0)
        self.assertIn("--exllamav3-root is deprecated and ignored", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
