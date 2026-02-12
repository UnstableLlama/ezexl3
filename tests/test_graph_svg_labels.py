import unittest

from ezexl3.graph_svg import _top_axis_ticks_and_labels


class GraphSvgTopAxisLabelTests(unittest.TestCase):
    def test_creates_six_inset_top_axis_gib_labels(self):
        ticks, labels = _top_axis_ticks_and_labels([2.0, 3.2, 4.8], tick_count=6, inset_ratio=0.07)

        self.assertEqual(len(ticks), 6)
        self.assertEqual(len(labels), 6)
        self.assertGreater(ticks[0], 2.0)
        self.assertLess(ticks[-1], 4.8)

    def test_clamps_tick_count_between_five_and_six(self):
        ticks_low, _ = _top_axis_ticks_and_labels([10.0, 16.0], tick_count=3)
        ticks_high, _ = _top_axis_ticks_and_labels([10.0, 16.0], tick_count=10)

        self.assertEqual(len(ticks_low), 5)
        self.assertEqual(len(ticks_high), 6)


if __name__ == "__main__":
    unittest.main()
