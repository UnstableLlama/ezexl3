import unittest

from ezexl3.graph_svg import _top_axis_ticks_and_labels


class GraphSvgTopAxisLabelTests(unittest.TestCase):
    def test_omits_rightmost_top_axis_gib_label(self):
        ticks, labels = _top_axis_ticks_and_labels([2.1, 3.2, 4.8])

        self.assertEqual(ticks, [2, 3, 4, 5])
        self.assertEqual(labels, ["2", "3", "4", ""])


if __name__ == "__main__":
    unittest.main()
