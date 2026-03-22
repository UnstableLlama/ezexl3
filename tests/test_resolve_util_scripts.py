import importlib.metadata
import json
import os
import tempfile
import types
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from ezexl3 import repo


def _make_util_scripts(root: Path) -> None:
    """Create dummy util/measure.py and util/optimize.py under *root*."""
    util = root / "util"
    util.mkdir(parents=True, exist_ok=True)
    (util / "measure.py").write_text("# measure")
    (util / "optimize.py").write_text("# optimize")


class ResolveUtilScriptsTests(unittest.TestCase):

    def test_exllamav3_root_env_takes_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_util_scripts(Path(tmp))
            with patch.dict(os.environ, {"EXLLAMAV3_ROOT": tmp}):
                measure, optimize = repo._resolve_exllamav3_util_scripts()
            self.assertEqual(measure, os.path.join(tmp, "util", "measure.py"))
            self.assertEqual(optimize, os.path.join(tmp, "util", "optimize.py"))

    def test_find_spec_parent_walk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_util_scripts(root)
            pkg_dir = root / "exllamav3"
            pkg_dir.mkdir()
            init = pkg_dir / "__init__.py"
            init.write_text("")

            fake_spec = types.SimpleNamespace(origin=str(init))
            with (
                patch.dict(os.environ, {}, clear=False),
                patch("ezexl3.repo.importlib.util.find_spec", return_value=fake_spec),
            ):
                os.environ.pop("EXLLAMAV3_ROOT", None)
                measure, optimize = repo._resolve_exllamav3_util_scripts()
            self.assertTrue(measure.endswith("util/measure.py"))
            self.assertTrue(os.path.isfile(measure))

    def test_direct_url_json_editable_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_util_scripts(root)

            direct_url = json.dumps({"url": f"file://{tmp}", "dir_info": {"editable": True}})
            mock_dist = MagicMock()
            mock_dist.read_text.return_value = direct_url
            mock_dist.files = None

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("ezexl3.repo.importlib.util.find_spec", return_value=None),
                patch("ezexl3.repo.importlib.metadata.distribution", return_value=mock_dist),
            ):
                os.environ.pop("EXLLAMAV3_ROOT", None)
                measure, optimize = repo._resolve_exllamav3_util_scripts()
            self.assertEqual(measure, os.path.join(tmp, "util", "measure.py"))
            self.assertEqual(optimize, os.path.join(tmp, "util", "optimize.py"))

    def test_installed_files_record_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_util_scripts(root)

            measure_rel = Path("util/measure.py")
            mock_dist = MagicMock()
            mock_dist.read_text.return_value = None
            mock_dist.files = [measure_rel]
            mock_dist.locate_file.return_value = root / "util" / "measure.py"

            with (
                patch.dict(os.environ, {}, clear=False),
                patch("ezexl3.repo.importlib.util.find_spec", return_value=None),
                patch("ezexl3.repo.importlib.metadata.distribution", return_value=mock_dist),
            ):
                os.environ.pop("EXLLAMAV3_ROOT", None)
                measure, optimize = repo._resolve_exllamav3_util_scripts()
            self.assertEqual(measure, str(root / "util" / "measure.py"))
            self.assertEqual(optimize, str(root / "util" / "optimize.py"))

    def test_raises_with_actionable_message_when_nothing_found(self):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch("ezexl3.repo.importlib.util.find_spec", return_value=None),
            patch(
                "ezexl3.repo.importlib.metadata.distribution",
                side_effect=repo.importlib.metadata.PackageNotFoundError("exllamav3"),
            ),
            patch(
                "ezexl3.repo.importlib.metadata.version",
                side_effect=repo.importlib.metadata.PackageNotFoundError("exllamav3"),
            ),
        ):
            os.environ.pop("EXLLAMAV3_ROOT", None)
            with self.assertRaises(RuntimeError) as ctx:
                repo._resolve_exllamav3_util_scripts()
            msg = str(ctx.exception)
            self.assertIn("EXLLAMAV3_ROOT", msg)
            self.assertIn("pip install -e", msg)


class DownloadUtilScriptsTests(unittest.TestCase):
    """Tests for strategy 5: auto-download from GitHub."""

    def _no_local_strategies(self):
        """Context manager that disables strategies 1-4 so download is reached."""
        mock_dist = MagicMock()
        mock_dist.read_text.return_value = None
        mock_dist.files = None
        return (
            patch.dict(os.environ, {}, clear=False),
            patch("ezexl3.repo.importlib.util.find_spec", return_value=None),
            patch("ezexl3.repo.importlib.metadata.distribution", return_value=mock_dist),
        )

    def test_download_caches_and_returns_scripts(self):
        with tempfile.TemporaryDirectory() as cache_root:
            p1, p2, p3 = self._no_local_strategies()

            def fake_urlopen(req, timeout=30):
                resp = MagicMock()
                resp.read.return_value = b"# script content"
                resp.__enter__ = lambda s: s
                resp.__exit__ = MagicMock(return_value=False)
                return resp

            with (
                p1, p2, p3,
                patch("ezexl3.repo.importlib.metadata.version", return_value="0.0.26"),
                patch("ezexl3.repo.importlib.metadata.metadata") as mock_meta,
                patch("urllib.request.urlopen", side_effect=fake_urlopen),
                patch.dict(os.environ, {"XDG_CACHE_HOME": cache_root}),
            ):
                os.environ.pop("EXLLAMAV3_ROOT", None)
                mock_meta.return_value = {"Home-page": "https://github.com/turboderp/exllamav3"}
                measure, optimize = repo._resolve_exllamav3_util_scripts()

            self.assertTrue(measure.endswith("measure.py"))
            self.assertTrue(optimize.endswith("optimize.py"))
            self.assertTrue(os.path.isfile(measure))
            self.assertTrue(os.path.isfile(optimize))
            self.assertIn("exllamav3-0.0.26", measure)

    def test_download_returns_cached_without_network(self):
        """Second call uses cached files, no network request."""
        with tempfile.TemporaryDirectory() as cache_root:
            cache_dir = Path(cache_root) / "ezexl3" / "exllamav3-1.0.0" / "util"
            cache_dir.mkdir(parents=True)
            (cache_dir / "measure.py").write_text("# cached measure")
            (cache_dir / "optimize.py").write_text("# cached optimize")

            attempted: list = []
            with (
                patch("ezexl3.repo.importlib.metadata.version", return_value="1.0.0"),
                patch("ezexl3.repo.importlib.metadata.metadata") as mock_meta,
                patch.dict(os.environ, {"XDG_CACHE_HOME": cache_root}),
            ):
                mock_meta.return_value = {"Home-page": "https://github.com/turboderp/exllamav3"}
                result = repo._download_exllamav3_util_scripts(attempted)

            self.assertIsNotNone(result)
            self.assertEqual(result[0], str(cache_dir / "measure.py"))
            self.assertEqual(result[1], str(cache_dir / "optimize.py"))

    def test_download_falls_back_to_version_without_v_prefix(self):
        """If v{version} tag 404s, tries {version} tag."""
        with tempfile.TemporaryDirectory() as cache_root:
            call_count = 0

            def fake_urlopen(req, timeout=30):
                nonlocal call_count
                call_count += 1
                url = req.full_url if hasattr(req, "full_url") else str(req)
                if "/v0.0.26/" in url:
                    raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
                resp = MagicMock()
                resp.read.return_value = b"# script"
                resp.__enter__ = lambda s: s
                resp.__exit__ = MagicMock(return_value=False)
                return resp

            attempted: list = []
            with (
                patch("ezexl3.repo.importlib.metadata.version", return_value="0.0.26"),
                patch("ezexl3.repo.importlib.metadata.metadata") as mock_meta,
                patch("urllib.request.urlopen", side_effect=fake_urlopen),
                patch.dict(os.environ, {"XDG_CACHE_HOME": cache_root}),
            ):
                mock_meta.return_value = {"Home-page": "https://github.com/turboderp/exllamav3"}
                result = repo._download_exllamav3_util_scripts(attempted)

            self.assertIsNotNone(result)
            # Should have attempted v0.0.26 first (failed), then 0.0.26 (succeeded)
            download_attempts = [a for a in attempted if a.startswith("(download)")]
            self.assertTrue(any("/v0.0.26/" in a for a in download_attempts))
            self.assertTrue(any("/0.0.26/" in a for a in download_attempts))

    def test_download_returns_none_on_total_failure(self):
        """If all download attempts fail, returns None."""
        attempted: list = []
        with (
            patch("ezexl3.repo.importlib.metadata.version", return_value="0.0.99"),
            patch("ezexl3.repo.importlib.metadata.metadata") as mock_meta,
            patch(
                "urllib.request.urlopen",
                side_effect=urllib.error.HTTPError("", 404, "Not Found", {}, None),
            ),
            patch.dict(os.environ, {"XDG_CACHE_HOME": "/tmp/nonexistent-cache-dir"}),
        ):
            mock_meta.return_value = {"Home-page": "https://github.com/turboderp/exllamav3"}
            result = repo._download_exllamav3_util_scripts(attempted)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
