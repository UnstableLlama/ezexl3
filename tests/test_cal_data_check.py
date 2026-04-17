import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, MagicMock

from ezexl3.quantize import _ensure_exl3_cal_data, _CAL_FILES, _CAL_BASE_URLS


def _install_fake_exl3_module(tmpdir):
    """
    Insert a fake exllamav3.conversion module into sys.modules
    whose __file__ points to tmpdir.  Returns a cleanup function.
    """
    fake_conversion = types.ModuleType("exllamav3.conversion")
    fake_conversion.__file__ = os.path.join(tmpdir, "__init__.py")

    fake_exl3 = types.ModuleType("exllamav3")
    fake_exl3.conversion = fake_conversion

    originals = {}
    for name in ("exllamav3", "exllamav3.conversion"):
        originals[name] = sys.modules.get(name)
    sys.modules["exllamav3"] = fake_exl3
    sys.modules["exllamav3.conversion"] = fake_conversion

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
                with patch("ezexl3.quantize.urllib.request.urlretrieve", side_effect=OSError("network error")), \
                     patch("ezexl3.quantize.time.sleep"):
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


    def test_falls_back_to_second_url(self):
        """If the first mirror 404s, the second mirror is tried."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cleanup = _install_fake_exl3_module(tmpdir)
            try:
                call_urls = []

                def fake_urlretrieve(url, dest):
                    call_urls.append(url)
                    if _CAL_BASE_URLS[0] in url:
                        raise OSError("404")
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    with open(dest, "w") as f:
                        f.write("data")

                with patch("ezexl3.quantize.urllib.request.urlretrieve", side_effect=fake_urlretrieve), \
                     patch("ezexl3.quantize.time.sleep"):
                    _ensure_exl3_cal_data()

                # Should have tried the first URL (3 retries) then succeeded on second
                first_mirror_calls = [u for u in call_urls if _CAL_BASE_URLS[0] in u]
                second_mirror_calls = [u for u in call_urls if _CAL_BASE_URLS[1] in u]
                self.assertGreater(len(first_mirror_calls), 0)
                self.assertGreater(len(second_mirror_calls), 0)
            finally:
                cleanup()


if __name__ == "__main__":
    unittest.main()
