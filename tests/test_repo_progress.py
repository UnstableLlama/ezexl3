import unittest

from ezexl3 import repo
from ezexl3 import repo_progress


class RepoProgressModuleTests(unittest.TestCase):
    def test_strip_ansi_removes_escape_sequences(self):
        self.assertEqual(repo_progress._strip_ansi("\x1b[32mhello\x1b[0m"), "hello")
        self.assertEqual(repo_progress._strip_ansi("\x1b[1;31mred bold\x1b[0m"), "red bold")

    def test_build_synthetic_bar_clamps_percentage(self):
        bar = repo_progress._build_synthetic_bar(120, width=10)
        self.assertEqual(bar, "━━━━━━━━━━ 100%")

        bar = repo_progress._build_synthetic_bar(-5, width=10)
        self.assertEqual(bar, "──────────   0%")

    def test_gpu_status_line_shrinks_progress_bar(self):
        text = "measure " + ("━" * 30) + " done"
        line = repo_progress._gpu_status_line(1, text, cols=30)

        self.assertTrue(line.startswith("\033[2K  GPU 1 | "))
        self.assertLessEqual(len(line.replace("\033[2K", "")), 30)
        self.assertIn("measure", line)
        self.assertIn("done", line)

    def test_repo_reexports_progress_helpers(self):
        self.assertIs(repo._strip_ansi, repo_progress._strip_ansi)
        self.assertIs(repo._gpu_status_line, repo_progress._gpu_status_line)
        self.assertIs(repo._build_synthetic_bar, repo_progress._build_synthetic_bar)
        self.assertIs(repo._clear_and_redraw_progress, repo_progress._clear_and_redraw_progress)
        self.assertIs(repo._print_above_progress, repo_progress._print_above_progress)
        self.assertIs(repo._init_gpu_progress, repo_progress._init_gpu_progress)
        self.assertIs(repo._redraw_gpu_progress, repo_progress._redraw_gpu_progress)
        self.assertIs(repo._print_msg_with_progress, repo_progress._print_msg_with_progress)
        self.assertIs(repo._cleanup_gpu_progress, repo_progress._cleanup_gpu_progress)


if __name__ == "__main__":
    unittest.main()
