import unittest

from ezexl3 import repo
from ezexl3 import repo_plan


class RepoPlanModuleTests(unittest.TestCase):
    def test_normalize_bpw_str_trims_integer_like_decimal(self):
        self.assertEqual(repo_plan._normalize_bpw_str("4.00"), "4")
        self.assertEqual(repo_plan._normalize_bpw_str("4.10"), "4.1")

    def test_dedupe_preserve_order_keeps_first_occurrence(self):
        self.assertEqual(repo_plan._dedupe_preserve_order(["4", "5", "4", "6", "5"]), ["4", "5", "6"])

    def test_plan_repo_bpws_opt_adds_integer_neighbors(self):
        plan = repo_plan._plan_repo_bpws(
            ["4", "4.07", "6.25"], opt_bpws={"4.07", "6.25"}
        )

        self.assertEqual(plan["requested_integers"], ["4"])
        self.assertEqual(plan["requested_optimizeds"], ["4.07", "6.25"])
        self.assertEqual(plan["quant_integer_queue"], ["4", "5", "6", "7"])
        # Optimized fracs are interleaved numerically in the measure queue.
        self.assertEqual(plan["measure_queue"], ["4", "4.07", "5", "6", "6.25", "7"])

    def test_plan_repo_bpws_without_opt_fractionals_go_direct(self):
        plan = repo_plan._plan_repo_bpws(["4", "4.07", "6.25"])

        self.assertEqual(plan["requested_integers"], ["4"])
        self.assertEqual(plan["requested_optimizeds"], [])
        self.assertEqual(plan["quant_integer_queue"], ["4", "4.07", "6.25"])
        self.assertEqual(plan["measure_queue"], ["4", "4.07", "6.25"])

    def test_plan_repo_bpws_sorts_fractionals_into_numeric_order(self):
        # User submits "-b 2,3,4,4.5,5,6,7" — 4.5 should run between 4 and 5,
        # not tacked onto the end after 7.
        plan = repo_plan._plan_repo_bpws(["2", "3", "4", "4.5", "5", "6", "7"])

        self.assertEqual(
            plan["quant_integer_queue"],
            ["2", "3", "4", "4.5", "5", "6", "7"],
        )
        self.assertEqual(
            plan["measure_queue"],
            ["2", "3", "4", "4.5", "5", "6", "7"],
        )

    def test_plan_repo_bpws_sorts_unsorted_input(self):
        # Input order shouldn't matter — the queue progresses monotonically.
        plan = repo_plan._plan_repo_bpws(["7", "4.5", "2", "5", "3"])

        self.assertEqual(
            plan["quant_integer_queue"],
            ["2", "3", "4.5", "5", "7"],
        )

    def test_repo_reexports_plan_helpers(self):
        self.assertIs(repo._normalize_bpw_str, repo_plan._normalize_bpw_str)
        self.assertIs(repo._split_integer_optimized_bpws, repo_plan._split_integer_optimized_bpws)
        self.assertIs(repo._dedupe_preserve_order, repo_plan._dedupe_preserve_order)
        self.assertIs(repo._plan_repo_bpws, repo_plan._plan_repo_bpws)


if __name__ == "__main__":
    unittest.main()
