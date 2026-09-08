import csv
import os
import tempfile
import unittest

from ezexl3.measure import (
    CSV_FIELDS,
    _lookup_csv_row,
    ensure_csv_exists,
    append_csv_row,
)


class LookupCsvRowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmp, "test.csv")
        ensure_csv_exists(self.csv_path)

    def test_returns_existing_row(self):
        row = {"weights": "4", "KL Div": "0.012", "PPL": "5.67", "GiB": "3.2"}
        append_csv_row(self.csv_path, row)

        result = _lookup_csv_row(self.csv_path, "4")
        self.assertEqual(result["weights"], "4")
        self.assertEqual(result["KL Div"], "0.012")

    def test_returns_empty_dict_for_missing_label(self):
        row = {"weights": "4", "KL Div": "0.012", "PPL": "5.67", "GiB": "3.2"}
        append_csv_row(self.csv_path, row)

        result = _lookup_csv_row(self.csv_path, "8")
        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()
