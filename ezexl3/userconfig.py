"""Persistent user config (~/.config/ezexl3/ui.json).

Shared by the chat server and the dashboard, which both read and write
this file and genuinely do run at the same time — the dashboard switch
spawns one from the other. That concurrency is the whole reason this
lives in one place: the two servers previously carried byte-identical
copies of these functions, and a non-atomic write in either could be
caught mid-truncate by a reader in the other, which then parsed {} and
persisted it back over the whole config.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

_LOCK = threading.Lock()


def config_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "ezexl3" / "ui.json"


def read_config() -> dict:
    """Strict read: {} only when the file genuinely isn't there.

    Raises on an existing-but-unparseable file, so a read-modify-write
    can tell "nothing saved yet" from "couldn't read what's saved" —
    conflating the two is how a whole config gets replaced by one key.
    """
    p = config_path()
    if not p.is_file():
        return {}
    return json.loads(p.read_text("utf-8"))


def load_config() -> dict:
    """Best-effort read for display. Never raises."""
    try:
        return read_config()
    except Exception:
        return {}


def save_config(data: dict) -> None:
    """Write atomically: temp file in the same dir, then rename over.

    A reader sees either the old file or the new one, never a partial.
    """
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=".ui.json.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def update_config(incoming: dict) -> None:
    """Merge incoming keys into the saved config and write it back."""
    with _LOCK:
        try:
            cfg = read_config()
        except Exception:
            # Corrupt or unreadable. Keep a copy instead of letting the
            # merge below silently overwrite it with near-nothing.
            p = config_path()
            try:
                p.replace(p.with_suffix(".json.corrupt"))
            except OSError:
                pass
            cfg = {}
        cfg.update(incoming)
        save_config(cfg)
