import io
import unittest
from contextlib import redirect_stderr

from ezexl3 import cli


class RemovedFlagTests(unittest.TestCase):
    """Verify that previously-deprecated flags are now fully removed."""

    def test_repo_rejects_exllamav3_root(self):
        argv = ["repo", "-m", "/tmp/model", "-b", "2", "--exllamav3-root", "/tmp/exl3"]
        with self.assertRaises(SystemExit):
            cli.main(argv)

    def test_repo_rejects_schedule(self):
        argv = ["repo", "-m", "/tmp/model", "-b", "2", "--schedule", "static"]
        with self.assertRaises(SystemExit):
            cli.main(argv)

    def test_repo_rejects_no_meta(self):
        argv = ["repo", "-m", "/tmp/model", "-b", "2", "--no-meta"]
        with self.assertRaises(SystemExit):
            cli.main(argv)

    def test_measure_rejects_exllamav3_root(self):
        argv = ["measure", "-m", "/tmp/model", "-b", "2", "--exllamav3-root", "/tmp/exl3"]
        with self.assertRaises(SystemExit):
            cli.main(argv)


if __name__ == "__main__":
    unittest.main()
