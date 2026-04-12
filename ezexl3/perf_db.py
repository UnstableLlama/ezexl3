# ezexl3/perf_db.py
"""
SQLite store for detailed per-context-length performance data.

Stores the full prefill and generation throughput curves produced by
eval/perf.py, keyed by BPW. The main measurement DB (measure_db.py)
keeps a single summary value per BPW; this module stores the complete
curve so the dashboard evals tab can render full tables per BPW.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Dict, List, Optional, Tuple

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS perf_results (
    bpw             TEXT    NOT NULL,
    phase           TEXT    NOT NULL,   -- 'prefill' or 'generation'
    context_length  INTEGER NOT NULL,
    tokens_per_second REAL  NOT NULL,
    PRIMARY KEY (bpw, phase, context_length)
);
"""

_BUSY_TIMEOUT_MS = 30_000
_RETRY_ATTEMPTS = 5
_RETRY_BASE_SLEEP = 0.1


def default_perf_db_path(model_dir: str) -> str:
    """Default perf database path: ``<model_dir>/<basename>PerfData.db``."""
    base = os.path.basename(os.path.abspath(model_dir.rstrip("/")))
    return os.path.join(model_dir, f"{base}PerfData.db")


def _connect(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=_BUSY_TIMEOUT_MS / 1000)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def upsert_perf_results(
    db_path: str,
    bpw: str,
    prefill: List[Tuple[int, float]],
    generation: List[Tuple[int, float]],
) -> None:
    """Write a full set of prefill + generation rows for one BPW.

    Each entry is ``(context_length, tokens_per_second)``.
    Existing rows for this BPW are replaced.
    """
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            conn = _connect(db_path)
            try:
                conn.execute(
                    "DELETE FROM perf_results WHERE bpw = ?", (bpw,)
                )
                for ctx, tps in prefill:
                    conn.execute(
                        "INSERT INTO perf_results (bpw, phase, context_length, tokens_per_second) "
                        "VALUES (?, 'prefill', ?, ?)",
                        (bpw, ctx, tps),
                    )
                for ctx, tps in generation:
                    conn.execute(
                        "INSERT INTO perf_results (bpw, phase, context_length, tokens_per_second) "
                        "VALUES (?, 'generation', ?, ?)",
                        (bpw, ctx, tps),
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


def read_perf_data(
    db_path: str, bpw: Optional[str] = None
) -> Dict[str, Dict[str, List[Dict]]]:
    """Read perf data, optionally filtered to one BPW.

    Returns ``{bpw: {"prefill": [...], "generation": [...]}}``.
    Each entry in the lists is ``{"context_length": int, "tokens_per_second": float}``.
    """
    if not os.path.exists(db_path):
        return {}

    conn = _connect(db_path)
    try:
        if bpw is not None:
            cur = conn.execute(
                "SELECT bpw, phase, context_length, tokens_per_second "
                "FROM perf_results WHERE bpw = ? ORDER BY phase, context_length",
                (bpw,),
            )
        else:
            cur = conn.execute(
                "SELECT bpw, phase, context_length, tokens_per_second "
                "FROM perf_results ORDER BY bpw, phase, context_length"
            )

        out: Dict[str, Dict[str, List[Dict]]] = {}
        for row_bpw, phase, ctx, tps in cur.fetchall():
            entry = out.setdefault(row_bpw, {"prefill": [], "generation": []})
            entry[phase].append({
                "context_length": ctx,
                "tokens_per_second": round(tps, 2),
            })
        return out
    finally:
        conn.close()


def available_bpws(db_path: str) -> List[str]:
    """Return sorted list of BPWs that have perf data."""
    if not os.path.exists(db_path):
        return []

    conn = _connect(db_path)
    try:
        cur = conn.execute(
            "SELECT DISTINCT bpw FROM perf_results ORDER BY bpw"
        )
        bpws = [row[0] for row in cur.fetchall()]
        # Sort numerically
        bpws.sort(key=lambda b: float(b) if b != "bf16" else -1.0)
        return bpws
    finally:
        conn.close()
