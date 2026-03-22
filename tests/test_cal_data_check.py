import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock

from ezexl3.quantize import _ensure_exl3_cal_data, _CAL_FILES


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

    def test_downloads_when_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cleanup = _install_fake_exl3_module(tmpdir)
            try:
                cal_dir = os.path.join(tmpdir, "standard_cal_data")

                def fake_urlretrieve(url, dest):
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "w") as f:
                        f.write(f"data for {os.path.basename(dest)}")

                with patch("ezexl3.quantize.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
                    _ensure_exl3_cal_data()

                for fname in _CAL_FILES:
                    path = os.path.join(cal_dir, fname)
                    self.assertTrue(os.path.isfile(path), f"{fname} should exist")
            finally:
                cleanup()

    def test_raises_on_download_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cleanup = _install_fake_exl3_module(tmpdir)
            try:
                with patch("ezexl3.quantize.urllib.request.urlretrieve", side_effect=OSError("network error")):
                    with self.assertRaises(RuntimeError) as ctx:
                        _ensure_exl3_cal_data()
                    self.assertIn("Failed to download", str(ctx.exception))
            finally:
                cleanup()

    def test_skips_already_downloaded_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cal_dir = os.path.join(tmpdir, "standard_cal_data")
            os.makedirs(cal_dir)
            # Create all but the first file
            for fname in _CAL_FILES[1:]:
                with open(os.path.join(cal_dir, fname), "w") as f:
                    f.write("existing")

            cleanup = _install_fake_exl3_module(tmpdir)
            try:
                downloaded = []

                def fake_urlretrieve(url, dest):
                    downloaded.append(os.path.basename(dest))
                    with open(dest, "w") as f:
                        f.write("new")

                with patch("ezexl3.quantize.urllib.request.urlretrieve", side_effect=fake_urlretrieve):
                    _ensure_exl3_cal_data()

                # Only the first file should have been downloaded
                self.assertEqual(downloaded, [_CAL_FILES[0]])
            finally:
                cleanup()


if __name__ == "__main__":
    unittest.main()
