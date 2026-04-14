"""
Regression test: every `from exllamav3 import <symbol>` statement in
non-vendor ezexl3 code must be guarded by a `try/except ImportError`
with a submodule fallback.

Some exllamav3 installs (e.g. partial / editable builds inside
containers) resolve the package as a PEP-420 namespace package, so the
top-level re-exports from `exllamav3/__init__.py` aren't available.  We
caught this once with `Config` (from `ppl_layer.py`) and again with
`model_init` (from `chat/inference.py`), so pin the pattern with a
static check.

Vendor files under `ezexl3/vendor/` are direct upstream copies and are
intentionally excluded — they get refreshed via VENDOR_MANIFEST.json.
"""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = REPO_ROOT / "ezexl3"
VENDOR_DIR = PKG_ROOT / "vendor"


def _find_python_files():
    for dirpath, _dirnames, filenames in os.walk(PKG_ROOT):
        p = Path(dirpath)
        # Skip vendor tree.
        try:
            p.relative_to(VENDOR_DIR)
            continue
        except ValueError:
            pass
        for fn in filenames:
            if fn.endswith(".py"):
                yield p / fn


def _ancestors(tree: ast.AST, target: ast.AST):
    """Yield ancestors of *target* in *tree*, outermost first."""
    stack = [(tree, [])]
    while stack:
        node, trail = stack.pop()
        if node is target:
            for a in trail:
                yield a
            return
        for child in ast.iter_child_nodes(node):
            stack.append((child, trail + [node]))


def _is_inside_guarded_try(tree: ast.AST, import_node: ast.ImportFrom) -> bool:
    """
    True if *import_node* is inside a `try` block whose `except` clauses
    catch ImportError (bare `except:` also counts as catching it).
    """
    for anc in _ancestors(tree, import_node):
        if isinstance(anc, ast.Try):
            for handler in anc.handlers:
                exc = handler.type
                if exc is None:
                    return True  # bare except
                # except ImportError
                if isinstance(exc, ast.Name) and exc.id == "ImportError":
                    return True
                # except (ImportError, ...)
                if isinstance(exc, ast.Tuple):
                    for elt in exc.elts:
                        if isinstance(elt, ast.Name) and elt.id == "ImportError":
                            return True
    return False


class ExllamaV3ImportsDefensive(unittest.TestCase):

    def test_top_level_exllamav3_imports_are_guarded(self):
        offenders: list[str] = []
        for path in _find_python_files():
            src = path.read_text(encoding="utf-8")
            try:
                tree = ast.parse(src, filename=str(path))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                # Only top-level `from exllamav3 import X` — submodule
                # imports like `from exllamav3.loader import …` are fine
                # because the submodules exist regardless of namespace
                # vs. regular-package resolution.
                if node.module != "exllamav3" or node.level:
                    continue
                if not _is_inside_guarded_try(tree, node):
                    rel = path.relative_to(REPO_ROOT)
                    names = ", ".join(a.name for a in node.names)
                    offenders.append(f"{rel}:{node.lineno}  from exllamav3 import {names}")

        if offenders:
            msg = (
                "Unguarded `from exllamav3 import ...` found. Wrap each in a\n"
                "try/except ImportError with a submodule fallback, e.g.:\n\n"
                "    try:\n"
                "        from exllamav3 import Config\n"
                "    except ImportError:\n"
                "        from exllamav3.model.config import Config\n\n"
                "Offenders:\n  " + "\n  ".join(offenders)
            )
            self.fail(msg)


if __name__ == "__main__":
    unittest.main()
