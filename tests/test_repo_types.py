import unittest

from ezexl3.repo_types import MeasureRuntimeConfig, RepoPlan


class RepoTypesTests(unittest.TestCase):
    def test_repo_plan_fields(self):
        plan = RepoPlan(
            requested_integers=["4"],
            requested_optimizeds=["4.07"],
            quant_integer_queue=["4", "5"],
            measure_queue=["4", "5", "4.07"],
        )

        self.assertEqual(plan.requested_integers, ["4"])
        self.assertEqual(plan.requested_optimizeds, ["4.07"])
        self.assertEqual(plan.quant_integer_queue, ["4", "5"])
        self.assertEqual(plan.measure_queue, ["4", "5", "4.07"])

    def test_measure_runtime_config_defaults(self):
        config = MeasureRuntimeConfig(
            model_dir="/tmp/model",
            devices=[0, 1],
            db_path="/tmp/model/measurements.db",
            out_csv="/tmp/model/measurements.csv",
            ppl_rows=128,
        )

        self.assertEqual(config.model_dir, "/tmp/model")
        self.assertEqual(config.devices, [0, 1])
        self.assertEqual(config.db_path, "/tmp/model/measurements.db")
        self.assertEqual(config.out_csv, "/tmp/model/measurements.csv")
        self.assertEqual(config.ppl_rows, 128)
        self.assertTrue(config.write_logs)


if __name__ == "__main__":
    unittest.main()
