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

CSV_CORE_FIELDS = ["weights", "KL Div", "PPL r-100", "GiB"]
CSV_EVAL_FIELDS = ["Diversity", "HumanEval", "IFBench", "LongCtx",
                   "MMLU", "Perf Prefill t/s", "Perf Gen t/s"]
# Legacy name kept for any external importers; describes the *full* column
# order when every eval is populated. The on-disk CSV is now dynamic —
# eval columns are only emitted when at least one row has a value.
CSV_FIELDS = CSV_CORE_FIELDS + CSV_EVAL_FIELDS

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS measurements (
    weights  TEXT PRIMARY KEY,
    kl_div   TEXT NOT NULL DEFAULT '',
    ppl      TEXT NOT NULL DEFAULT '',
    gib      TEXT NOT NULL DEFAULT ''
);
"""

# Eval columns added via migration (ALTER TABLE) so existing DBs upgrade seamlessly.
_EVAL_COLUMNS = [
    "diversity_score",
    "humaneval_pass",
    "ifbench_score",
    "longctx_score",
    "mmlu_accuracy",
    "perf_prefill_tps",
    "perf_gen_tps",
]

# Maps DB column names to CSV header names.
_EVAL_COL_TO_CSV = {
    "diversity_score": "Diversity",
    "humaneval_pass": "HumanEval",
    "ifbench_score": "IFBench",
    "longctx_score": "LongCtx",
    "mmlu_accuracy": "MMLU",
    "perf_prefill_tps": "Perf Prefill t/s",
    "perf_gen_tps": "Perf Gen t/s",
}

# Retry / busy-timeout for concurrent writers
_BUSY_TIMEOUT_MS = 30_000
_RETRY_ATTEMPTS = 5
_RETRY_BASE_SLEEP = 0.1  # seconds


def _db_path(model_dir: str) -> str:
    """Default database path: ``<model_dir>/<basename>Measured.db``."""
    base = os.path.basename(os.path.abspath(model_dir.rstrip("/")))
    return os.path.join(model_dir, f"{base}Measured.db")


def _migrate_eval_columns(conn: sqlite3.Connection) -> None:
    """Add eval columns to an existing measurements table (idempotent)."""
    for col in _EVAL_COLUMNS:
        try:
            conn.execute(f"ALTER TABLE measurements ADD COLUMN {col} TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute(_SCHEMA)
    conn.commit()
    _migrate_eval_columns(conn)
    return conn


def default_db_path(model_dir: str) -> str:
    return _db_path(model_dir)


def upsert_row(
    db_path: str,
    weights: str,
    kl_div: str = "",
    ppl: str = "",
    gib: str = "",
    **eval_kwargs: str,
) -> None:
    """Insert or field-level update a measurement row.

    Only non-empty values overwrite existing data, so a KL-only write
    won't blank out an earlier PPL value for the same label.

    Extra keyword arguments whose names match ``_EVAL_COLUMNS`` are
    persisted in the corresponding eval columns.
    """
    # Build the base columns + any eval columns provided.
    cols = ["weights", "kl_div", "ppl", "gib"]
    vals: List[str] = [weights, str(kl_div), str(ppl), str(gib)]
    for col in _EVAL_COLUMNS:
        v = str(eval_kwargs.get(col, ""))
        cols.append(col)
        vals.append(v)

    placeholders = ", ".join("?" for _ in cols)
    col_names = ", ".join(cols)
    # Build the ON CONFLICT SET clause — skip 'weights' (the PK).
    set_clauses = []
    for c in cols[1:]:
        set_clauses.append(
            f"{c} = CASE WHEN excluded.{c} != '' THEN excluded.{c} ELSE measurements.{c} END"
        )
    set_sql = ", ".join(set_clauses)

    sql = (
        f"INSERT INTO measurements ({col_names}) VALUES ({placeholders}) "
        f"ON CONFLICT(weights) DO UPDATE SET {set_sql}"
    )

    for attempt in range(_RETRY_ATTEMPTS):
        try:
            conn = _connect(db_path)
            try:
                conn.execute(sql, tuple(vals))
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
        eval_col_sql = ", ".join(_EVAL_COLUMNS)
        cur = conn.execute(
            f"SELECT weights, kl_div, ppl, gib, {eval_col_sql} FROM measurements ORDER BY weights"
        )
        out: Dict[str, dict] = {}
        for row in cur.fetchall():
            weights, kl_div, ppl, gib = row[0], row[1], row[2], row[3]
            d: dict = {
                "weights": weights,
                "KL Div": kl_div,
                "PPL r-100": ppl,
                "GiB": gib,
            }
            for i, col in enumerate(_EVAL_COLUMNS):
                d[_EVAL_COL_TO_CSV[col]] = row[4 + i]
            out[weights] = d
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
    """Write the database contents to a CSV file (sorted by BPW).

    Columns are emitted dynamically: the 4 core columns
    (``weights``, ``KL Div``, ``PPL r-100``, ``GiB``) are always
    included. Each eval column is included only if at least one row
    has a non-empty value for it, so a baseline KL/PPL/GiB run
    produces a clean 4-column CSV while hidden-flag runs that set
    Diversity / MMLU / Perf / etc. keep those values in the output.
    """
    rows = read_all_rows(db_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    # Pick up each eval column only when some row actually populated it.
    active_eval_fields = [
        field for field in CSV_EVAL_FIELDS
        if any((row.get(field) or "").strip() for row in rows.values())
    ]
    fieldnames = CSV_CORE_FIELDS + active_eval_fields

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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
