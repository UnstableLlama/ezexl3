import csv
import tempfile
import unittest
from pathlib import Path

from ezexl3.graph_svg import generate_iceblink_svg


class GraphSvgTopAxisLabelTests(unittest.TestCase):
    def test_omits_rightmost_top_axis_gib_label(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            csv_path = td_path / "Measured.csv"
            out_svg = td_path / "plot.svg"

            with csv_path.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["weights", "KL Div", "PPL r-100", "GiB"])
                w.writeheader()
                w.writerow({"weights": "2", "KL Div": "0.2", "PPL r-100": "12.0", "GiB": "2.0"})
                w.writerow({"weights": "3", "KL Div": "0.1", "PPL r-100": "11.0", "GiB": "3.0"})
                w.writerow({"weights": "4", "KL Div": "0.05", "PPL r-100": "10.5", "GiB": "4.0"})

            generate_iceblink_svg(str(csv_path), str(out_svg), "test")
            svg = out_svg.read_text()

            self.assertIn(">2.00</text>", svg)
            self.assertIn(">3.00</text>", svg)
            self.assertNotIn(">4.00</text>", svg)


if __name__ == "__main__":
    unittest.main()
