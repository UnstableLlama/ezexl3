import json
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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
        ):
            os.environ.pop("EXLLAMAV3_ROOT", None)
            with self.assertRaises(RuntimeError) as ctx:
                repo._resolve_exllamav3_util_scripts()
            msg = str(ctx.exception)
            self.assertIn("EXLLAMAV3_ROOT", msg)
            self.assertIn("pip install -e", msg)


if __name__ == "__main__":
    unittest.main()
