# ezexl3/measure_db.py
"""
SQLite-backed measurement store.

Replaces the old shard-CSV-plus-merge pattern with a single WAL-mode
SQLite database.  Multiple GPU worker processes can safely upsert rows
concurrently — SQLite serialises the (tiny) writes automatically.

The CSV that downstream code (readme, graph_svg) expects is exported
once at the end via ``export_csv()``.
"""

from __future__ import annotations

import csv
import math
import os
import sqlite3
import time
from typing import Dict, List, Optional, Set

CSV_FIELDS = ["weights", "KL Div", "PPL r-100", "GiB"]

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS measurements (
    weights  TEXT PRIMARY KEY,
    kl_div   TEXT NOT NULL DEFAULT '',
    ppl      TEXT NOT NULL DEFAULT '',
    gib      TEXT NOT NULL DEFAULT ''
);
"""

# Retry / busy-timeout for concurrent writers
_BUSY_TIMEOUT_MS = 30_000
_RETRY_ATTEMPTS = 5
_RETRY_BASE_SLEEP = 0.1  # seconds


def _db_path(model_dir: str) -> str:
    """Default database path: ``<model_dir>/<basename>Measured.db``."""
    base = os.path.basename(os.path.abspath(model_dir.rstrip("/")))
    return os.path.join(model_dir, f"{base}Measured.db")


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def default_db_path(model_dir: str) -> str:
    return _db_path(model_dir)


def upsert_row(
    db_path: str,
    weights: str,
    kl_div: str = "",
    ppl: str = "",
    gib: str = "",
) -> None:
    """Insert or field-level update a measurement row.

    Only non-empty values overwrite existing data, so a KL-only write
    won't blank out an earlier PPL value for the same label.
    """
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            conn = _connect(db_path)
            try:
                conn.execute(
                    """\
                    INSERT INTO measurements (weights, kl_div, ppl, gib)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(weights) DO UPDATE SET
                        kl_div = CASE WHEN excluded.kl_div != '' THEN excluded.kl_div ELSE measurements.kl_div END,
                        ppl    = CASE WHEN excluded.ppl    != '' THEN excluded.ppl    ELSE measurements.ppl    END,
                        gib    = CASE WHEN excluded.gib    != '' THEN excluded.gib    ELSE measurements.gib    END
                    """,
                    (weights, str(kl_div), str(ppl), str(gib)),
                )
                conn.commit()
            finally:
                conn.close()
            return
        except sqlite3.OperationalError:
            if attempt < _RETRY_ATTEMPTS - 1:
                time.sleep(_RETRY_BASE_SLEEP * (2 ** attempt))
            else:
                raise


def read_all_rows(db_path: str) -> Dict[str, dict]:
    """Return ``{label: {"weights": ..., "KL Div": ..., ...}}`` for every row."""
    if not os.path.exists(db_path):
        return {}
    conn = _connect(db_path)
    try:
        cur = conn.execute("SELECT weights, kl_div, ppl, gib FROM measurements ORDER BY weights")
        out: Dict[str, dict] = {}
        for weights, kl_div, ppl, gib in cur.fetchall():
            out[weights] = {
                "weights": weights,
                "KL Div": kl_div,
                "PPL r-100": ppl,
                "GiB": gib,
            }
        return out
    finally:
        conn.close()


def read_existing_weights(db_path: str) -> Set[str]:
    return set(read_all_rows(db_path).keys())


def _bpw_sort_key(label: str) -> float:
    """Sort labels numerically where possible, text labels last."""
    try:
        return float(label)
    except ValueError:
        return math.inf


def export_csv(db_path: str, csv_path: str) -> None:
    """Write the database contents to a CSV file (sorted by BPW)."""
    rows = read_all_rows(db_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for key in sorted(rows.keys(), key=_bpw_sort_key):
            w.writerow(rows[key])
        f.flush()
        os.fsync(f.fileno())


def migrate_csv_to_db(csv_path: str, db_path: str) -> int:
    """Import rows from an existing CSV into the database (for resume support).

    Returns the number of rows imported.
    """
    if not os.path.exists(csv_path):
        return 0
    count = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get("weights") or "").strip()
            if not w:
                continue
            upsert_row(
                db_path,
                weights=w,
                kl_div=(row.get("KL Div") or "").strip(),
                ppl=(row.get("PPL r-100") or "").strip(),
                gib=(row.get("GiB") or "").strip(),
            )
            count += 1
    return count
