import importlib
import unittest

from ezexl3 import repo


class RepoMeasureModuleTests(unittest.TestCase):
    def test_parse_measure_args_matches_repo_wrapper(self):
        repo_measure = importlib.import_module("ezexl3.repo_measure")
        args = ["-r", "200", "-d", "2,3"]

        self.assertEqual(
            repo_measure._parse_measure_args(args, [0]),
            repo._parse_measure_args(args, [0]),
        )

    def test_task_to_csv_label_and_checkpoint_filter_match_repo_wrapper(self):
        repo_measure = importlib.import_module("ezexl3.repo_measure")

        self.assertEqual(repo_measure._task_to_csv_label("base"), repo._task_to_csv_label("base"))
        self.assertEqual(
            repo_measure._filter_measure_tasks_for_checkpoint(["2", "base", "3"], {"bf16", "3"}),
            repo._filter_measure_tasks_for_checkpoint(["2", "base", "3"], {"bf16", "3"}),
        )

    def test_init_measure_db_matches_repo_wrapper(self):
        repo_measure = importlib.import_module("ezexl3.repo_measure")
        self.assertEqual(
            repo_measure._init_measure_db("/tmp/model", [0, 1]),
            repo._init_measure_db("/tmp/model", [0, 1]),
        )


if __name__ == "__main__":
    unittest.main()
