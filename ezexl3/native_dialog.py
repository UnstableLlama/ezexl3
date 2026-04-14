# ezexl3/native_dialog.py
"""
Native OS directory picker dialog.

Uses tkinter's file dialog to open the platform's native directory
chooser.  Falls back gracefully when tkinter or a display is
unavailable (headless servers, Docker containers, etc.).
"""

from __future__ import annotations

import os
from typing import Optional


def pick_directory(initial_dir: str = "") -> Optional[str]:
    """Open a native directory picker and return the chosen path.

    Returns *None* if the user cancels or if no display is available.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    try:
        root = tk.Tk()
        root.withdraw()
        # Bring the dialog above the browser window.
        root.attributes("-topmost", True)

        kwargs: dict = {"title": "Select Directory"}
        if initial_dir and os.path.isdir(initial_dir):
            kwargs["initialdir"] = initial_dir

        path = filedialog.askdirectory(**kwargs)
        root.destroy()
        return path or None
    except Exception:
        # No display, Wayland issues, etc.
        return None
