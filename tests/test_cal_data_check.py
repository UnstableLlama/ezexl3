import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock

from ezexl3.quantize import _ensure_exl3_cal_data, _find_cal_data_source


def _install_fake_exl3_module(tmpdir):
    """
    Insert a fake exllamav3.conversion.calibration_data module into sys.modules
    whose __file__ points to tmpdir.  Returns a cleanup function.
    """
    fake_cd = types.ModuleType("exllamav3.conversion.calibration_data")
    fake_cd.__file__ = os.path.join(tmpdir, "calibration_data.py")

    fake_conversion = types.ModuleType("exllamav3.conversion")
    fake_conversion.calibration_data = fake_cd

    fake_exl3 = types.ModuleType("exllamav3")
    fake_exl3.conversion = fake_conversion

    originals = {}
    for name in ("exllamav3", "exllamav3.conversion", "exllamav3.conversion.calibration_data"):
        originals[name] = sys.modules.get(name)
    sys.modules["exllamav3"] = fake_exl3
    sys.modules["exllamav3.conversion"] = fake_conversion
    sys.modules["exllamav3.conversion.calibration_data"] = fake_cd

    def cleanup():
        for name, orig in originals.items():
            if orig is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = orig

    return cleanup


class EnsureCalDataTests(unittest.TestCase):
    """Tests for calibration-data preflight check."""

    def test_passes_when_c4_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_dir = os.path.join(tmpdir, "standard_cal_data")
            os.makedirs(cal_dir)
            with open(os.path.join(cal_dir, "c4.utf8"), "w") as f:
                f.write("test data")

            cleanup = _install_fake_exl3_module(tmpdir)
            try:
                _ensure_exl3_cal_data()  # should not raise
            finally:
                cleanup()

    def test_raises_when_c4_missing_and_no_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cleanup = _install_fake_exl3_module(tmpdir)
            try:
                with patch("ezexl3.quantize._find_cal_data_source", return_value=None):
                    with self.assertRaises(RuntimeError) as ctx:
                        _ensure_exl3_cal_data()
                    self.assertIn("c4.utf8", str(ctx.exception))
            finally:
                cleanup()

    def test_auto_repairs_from_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Fake source with c4.utf8
            source_dir = os.path.join(tmpdir, "source")
            os.makedirs(source_dir)
            source_c4 = os.path.join(source_dir, "c4.utf8")
            with open(source_c4, "w") as f:
                f.write("calibration data")

            # Fake installed package (missing cal data)
            pkg_dir = os.path.join(tmpdir, "installed")
            os.makedirs(pkg_dir)

            cleanup = _install_fake_exl3_module(pkg_dir)
            try:
                with patch("ezexl3.quantize._find_cal_data_source", return_value=source_c4):
                    _ensure_exl3_cal_data()
            finally:
                cleanup()

            repaired = os.path.join(pkg_dir, "standard_cal_data", "c4.utf8")
            self.assertTrue(os.path.isfile(repaired))
            with open(repaired) as f:
                self.assertEqual(f.read(), "calibration data")


class FindCalDataSourceTests(unittest.TestCase):
    """Tests for _find_cal_data_source metadata lookup."""

    def test_returns_none_when_no_metadata(self):
        with patch("importlib.metadata.distribution", side_effect=Exception):
            self.assertIsNone(_find_cal_data_source())

    def test_returns_path_for_editable_install(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_dir = os.path.join(
                tmpdir, "exllamav3", "conversion", "standard_cal_data",
            )
            os.makedirs(cal_dir)
            c4_path = os.path.join(cal_dir, "c4.utf8")
            with open(c4_path, "w") as f:
                f.write("data")

            direct_url = json.dumps({"url": f"file://{tmpdir}"})

            class FakeDist:
                def read_text(self, name):
                    if name == "direct_url.json":
                        return direct_url
                    return None

            with patch("importlib.metadata.distribution", return_value=FakeDist()):
                result = _find_cal_data_source()

            self.assertEqual(result, c4_path)

    def test_returns_none_for_non_file_url(self):
        direct_url = json.dumps({"url": "https://example.com/exllamav3"})

        class FakeDist:
            def read_text(self, name):
                if name == "direct_url.json":
                    return direct_url
                return None

        with patch("importlib.metadata.distribution", return_value=FakeDist()):
            self.assertIsNone(_find_cal_data_source())


if __name__ == "__main__":
    unittest.main()
