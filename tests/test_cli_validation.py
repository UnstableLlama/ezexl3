import unittest

from ezexl3 import cli


class CliValidationTests(unittest.TestCase):
    def test_parse_devices_rejects_non_integer(self):
        with self.assertRaises(SystemExit):
            cli._parse_devices(["0", "gpu1"])

    def test_parse_devices_rejects_empty(self):
        with self.assertRaises(SystemExit):
            cli._parse_devices([])

    def test_parse_device_ratios_rejects_length_mismatch(self):
        with self.assertRaises(SystemExit):
            cli._parse_device_ratios(["1"], [0, 1])

    def test_parse_device_ratios_rejects_non_positive(self):
        with self.assertRaises(SystemExit):
            cli._parse_device_ratios(["1", "0"], [0, 1])

    def test_parse_device_ratios_accepts_valid(self):
        out = cli._parse_device_ratios(["1", "1.5"], [0, 1])
        self.assertEqual(out, ["1", "1.5"])


if __name__ == "__main__":
    unittest.main()
