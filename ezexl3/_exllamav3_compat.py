"""Compatibility shims for older exllamav3 installs.

Our vendored scripts and internal modules track current upstream exllamav3,
which places ``ProgressBar`` in ``exllamav3.util.progress``. Older releases
kept it elsewhere — typically re-exported from ``exllamav3.util`` via a
wildcard import from ``exllamav3.util.misc``.

``install_progress_shim()`` synthesizes the ``exllamav3.util.progress``
submodule when it is missing, forwarding to whichever legacy location
still holds ``ProgressBar``. No-op on installs where the real submodule
already exists. Idempotent.
"""

from __future__ import annotations


def install_progress_shim() -> None:
    try:
        import exllamav3.util.progress  # noqa: F401
        return
    except ImportError:
        pass

    progress_bar = None
    try:
        from exllamav3.util import ProgressBar as progress_bar  # type: ignore  # noqa: F401
    except ImportError:
        try:
            from exllamav3.util.misc import ProgressBar as progress_bar  # type: ignore  # noqa: F401
        except ImportError:
            return

    import sys
    import types

    shim = types.ModuleType("exllamav3.util.progress")
    shim.ProgressBar = progress_bar
    sys.modules["exllamav3.util.progress"] = shim
