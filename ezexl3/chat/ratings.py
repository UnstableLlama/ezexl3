# Preference-rating store for the chat UI (KTO / DPO data collection).
#
# The UI captures in one of two modes (header toggle; a third "Off"
# position — the default — disables capture entirely):
#   KTO — 👍/👎 on a reply writes one independent labeled row.
#   DPO — each send/regen produces two candidates; marking one ▲ chosen
#         and one ▼ rejected then committing writes one pair.
#
# Rows are appended to plain JSONL files in the exact column format that
# UnstableLlama/exllamav3's training/qlora_train_pref.py reads with its
# default keys, so a collected dataset trains with zero conversion:
#
#   <dataset>.kto.jsonl : {"prompt": [...], "completion": str, "label": bool}
#   <dataset>.dpo.jsonl : {"prompt": [...], "chosen": str, "rejected": str}
#
# "prompt" is a TRL-conversational list of {role, content} turns (full chat
# history; the trainer currently folds this to system + last user turn, but
# the stored data keeps the full context for future multi-turn training).
# Extra provenance keys (node_id, model, ts, source) ride along on every row;
# the trainer selects columns by name and ignores them.

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

# Dataset names become file names — keep them boring.
_DATASET_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

_ROLES = {"system", "user", "assistant"}

# Serializes read-modify-write cycles across aiohttp's to_thread workers.
_LOCK = threading.Lock()


def default_datasets_dir() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME", "")
    base = Path(xdg) if xdg else Path.home() / ".local" / "share"
    return base / "ezexl3" / "preference_data"


def valid_dataset_name(name: str) -> bool:
    return bool(_DATASET_RE.fullmatch(name or ""))


def validate_prompt(prompt) -> str | None:
    """Return an error string, or None if *prompt* is a valid turn list."""
    if not isinstance(prompt, list) or not prompt:
        return "prompt must be a non-empty list of {role, content} turns"
    for turn in prompt:
        if not isinstance(turn, dict):
            return "each prompt turn must be an object"
        if turn.get("role") not in _ROLES:
            return f"invalid turn role: {turn.get('role')!r}"
        if not isinstance(turn.get("content"), str):
            return "each prompt turn needs string content"
    if prompt[-1].get("role") != "user":
        return "final prompt turn must be from the user"
    return None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RatingsStore:
    """JSONL-backed store; one .kto.jsonl + one .dpo.jsonl per dataset.

    Files are small (human-scale ratings), so mutations do an atomic
    read-filter-append-rewrite. Lines that don't parse as JSON objects
    (e.g. hand-edited) are preserved verbatim and never rewritten.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    # ── paths / io ────────────────────────────────────────────────

    def _path(self, dataset: str, kind: str) -> Path:
        if not valid_dataset_name(dataset):
            raise ValueError(f"invalid dataset name: {dataset!r}")
        return self.root / f"{dataset}.{kind}.jsonl"

    @staticmethod
    def _read(path: Path) -> list:
        """Returns a list of dicts (parsed rows) and raw strings (kept lines)."""
        if not path.is_file():
            return []
        out = []
        for line in path.read_text("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                out.append(row if isinstance(row, dict) else line)
            except json.JSONDecodeError:
                out.append(line)
        return out

    @staticmethod
    def _write(path: Path, rows: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                if isinstance(row, str):
                    f.write(row + "\n")
                else:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(tmp, path)

    # ── mutations ─────────────────────────────────────────────────

    def rate_kto(self, dataset: str, node_id: str, prompt: list,
                 completion: str, label: bool | None, model: str) -> None:
        """Upsert one KTO row keyed by node_id; label None removes it."""
        path = self._path(dataset, "kto")
        with _LOCK:
            rows = [r for r in self._read(path)
                    if isinstance(r, str) or r.get("node_id") != node_id]
            if label is not None:
                rows.append({
                    "prompt": prompt,
                    "completion": completion,
                    "label": bool(label),
                    "node_id": node_id,
                    "model": model,
                    "ts": _now(),
                })
            self._write(path, rows)

    def rate_dpo_pair(self, dataset: str, prompt: list, chosen: dict,
                      rejected: dict, model: str,
                      remove: bool = False) -> None:
        """Upsert (or with remove=True, delete) the DPO pair for one duo.

        The pair is keyed by the UNORDERED {chosen, rejected} node-id duo,
        so re-picking the other candidate of a duel replaces the old row
        instead of leaving two contradictory pairs on disk.
        """
        path = self._path(dataset, "dpo")
        duo = {chosen["node_id"], rejected["node_id"]}
        with _LOCK:
            rows = [r for r in self._read(path)
                    if isinstance(r, str)
                    or {r.get("chosen_node_id"),
                        r.get("rejected_node_id")} != duo]
            if not remove:
                rows.append({
                    "prompt": prompt,
                    "chosen": chosen["content"],
                    "rejected": rejected["content"],
                    "chosen_node_id": chosen["node_id"],
                    "rejected_node_id": rejected["node_id"],
                    "source": "duel",
                    "model": model,
                    "ts": _now(),
                })
            self._write(path, rows)

    # ── queries ───────────────────────────────────────────────────

    def state(self, dataset: str) -> dict:
        """UI-facing state: node-keyed labels + pair ids (no content)."""
        kto = {}
        for r in self._read(self._path(dataset, "kto")):
            if isinstance(r, dict) and r.get("node_id"):
                kto[r["node_id"]] = bool(r.get("label"))
        dpo = []
        for r in self._read(self._path(dataset, "dpo")):
            if isinstance(r, dict) and r.get("chosen_node_id"):
                dpo.append({
                    "chosen": r["chosen_node_id"],
                    "rejected": r.get("rejected_node_id"),
                    "source": r.get("source", "manual"),
                })
        return {"kto": kto, "dpo": dpo}

    def list_datasets(self) -> list[str]:
        if not self.root.is_dir():
            return []
        names = set()
        for p in self.root.iterdir():
            for kind in (".kto.jsonl", ".dpo.jsonl"):
                if p.name.endswith(kind):
                    names.add(p.name[: -len(kind)])
        return sorted(names)
