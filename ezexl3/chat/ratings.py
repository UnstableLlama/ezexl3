# Dataset store for the chat UI (Single / KTO / DPO data collection).
#
# The UI captures in one of three modes (header toggle; an "Off"
# position — the default — disables capture entirely):
#   Single — a reply is saved as one plain conversation row (SFT data):
#            💾 on any reply, or every kept reply of a queue run.
#   KTO — 👍/👎 on a reply writes one independent labeled row.
#   DPO — each send/regen produces two candidates; marking one ▲ chosen
#         and one ▼ rejected then committing writes one pair.
#
# Rows are appended to plain JSONL files in the exact column format the
# UnstableLlama/exllamav3 fork's trainers read with their default keys, so
# a collected dataset trains with zero conversion:
#
#   <dataset>.single.jsonl : {"messages": [...turns..., assistant turn]}
#                            (SFT: qlora_train*.py --messages-key messages)
#   <dataset>.kto.jsonl    : {"prompt": [...], "completion": str, "label": bool}
#   <dataset>.dpo.jsonl    : {"prompt": [...], "chosen": str, "rejected": str}
#                            (training/qlora_train_pref.py)
#
# "prompt" / "messages" are TRL-conversational lists of {role, content}
# turns (full chat history; the preference trainer currently folds this to
# system + last user turn, but the stored data keeps the full context for
# future multi-turn training). Extra provenance keys (node_id, model, ts,
# source, and the generation prompts — chosen_system/rejected_system on
# DPO rows, gen_system on Single rows) ride along on every row; the
# trainers select columns by name and ignore them.

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


# Thought blocks stripped from completions bound for a dataset when the
# strip-thinking option is on: gemma4 <|channel>…<channel|> and generic
# <think>…</think>. Mirrors stripThink() in static/js/ratings.js.
_THINK_RES = (
    re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL),
    re.compile(r"<think>.*?</think>", re.DOTALL),
)


def strip_think_text(text: str) -> str:
    """Remove thought blocks (tags included) and leading whitespace."""
    if not text:
        return text
    for rx in _THINK_RES:
        text = rx.sub("", text)
    return text.lstrip()


class RatingsStore:
    """JSONL-backed store; .single/.kto/.dpo.jsonl files per dataset.

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

        chosen/rejected may carry "gen_system" — the system prompt that
        candidate was actually generated under, when it differed from the
        trained prompt stored in the "prompt" column. It lands in the
        chosen_system/rejected_system metadata columns (null when the
        trained prompt was used); the trainer ignores them.
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
                    "chosen_system": chosen.get("gen_system"),
                    "rejected_system": rejected.get("gen_system"),
                    "source": "duel",
                    "model": model,
                    "ts": _now(),
                })
            self._write(path, rows)

    def add_bulk_rows(self, dataset: str, prompt: list, completions: list,
                      target: str, model: str,
                      source_row: dict | None = None,
                      gen_system: str | None = None) -> int:
        """Append one DPO row per generated completion (bulk generation).

        *target* names the side the generated text fills ("chosen" or
        "rejected"). The opposite column comes from *source_row* — the
        loaded dataset row this prompt came from, e.g. {"chosen": str,
        "rejected": str, "id": str} — or "" when absent, leaving a
        half-pair to be joined later. *completions* is a list of
        {"content": str, "node_id": str|None}; rows carrying a node id
        (review-flow saves) upsert — any prior row referencing that node
        is replaced. *gen_system* is the generation-time system prompt,
        recorded as metadata on the generated side only.

        Returns the number of rows written.
        """
        if target not in ("chosen", "rejected"):
            raise ValueError(f"invalid bulk target: {target!r}")
        other = "rejected" if target == "chosen" else "chosen"
        src = source_row or {}
        other_text = src.get(other) if isinstance(src.get(other), str) else ""
        source_row_id = src.get("id")

        path = self._path(dataset, "dpo")
        ids = {c["node_id"] for c in completions if c.get("node_id")}
        with _LOCK:
            rows = [r for r in self._read(path)
                    if isinstance(r, str)
                    or not ({r.get("chosen_node_id"),
                             r.get("rejected_node_id")} & ids)]
            written = 0
            for c in completions:
                content = c.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                row = {
                    "prompt": prompt,
                    "chosen": content if target == "chosen" else other_text,
                    "rejected": content if target == "rejected" else other_text,
                    "chosen_node_id": c.get("node_id") if target == "chosen" else None,
                    "rejected_node_id": c.get("node_id") if target == "rejected" else None,
                    "chosen_system": gen_system if target == "chosen" else None,
                    "rejected_system": gen_system if target == "rejected" else None,
                    "source": "bulk",
                    "source_row_id": source_row_id,
                    "model": model,
                    "ts": _now(),
                }
                rows.append(row)
                written += 1
            self._write(path, rows)
        return written

    def add_single_rows(self, dataset: str, prompt: list, completions: list,
                        model: str, source: str = "bulk",
                        source_row_id=None,
                        gen_system: str | None = None) -> int:
        """Append one plain conversation row per completion (Single mode).

        Each row is {"messages": prompt + [assistant turn]} — the OpenAI
        layout the fork's SFT trainers read with --messages-key messages
        — plus provenance. *completions* is a list of {"content": str,
        "node_id": str|None}; rows carrying a node id upsert (re-saving
        the same reply replaces its row). *source* is "manual" (💾 on a
        reply) or "bulk" (queue runs, reviewed or unattended).
        *gen_system* is the generation-time system prompt when it
        differed from the saved system turn — metadata only, the trainer
        ignores it. Returns the number of rows written.
        """
        if source not in ("manual", "bulk"):
            raise ValueError(f"invalid single source: {source!r}")
        path = self._path(dataset, "single")
        ids = {c["node_id"] for c in completions if c.get("node_id")}
        with _LOCK:
            rows = [r for r in self._read(path)
                    if isinstance(r, str) or r.get("node_id") not in ids]
            written = 0
            for c in completions:
                content = c.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                rows.append({
                    "messages": list(prompt)
                                + [{"role": "assistant", "content": content}],
                    "node_id": c.get("node_id"),
                    "gen_system": gen_system,
                    "source": source,
                    "source_row_id": source_row_id,
                    "model": model,
                    "ts": _now(),
                })
                written += 1
            self._write(path, rows)
        return written

    def remove_single_rows(self, dataset: str, node_ids: list) -> None:
        """Delete the Single rows keyed by any of *node_ids* (💾 withdraw)."""
        path = self._path(dataset, "single")
        drop = {n for n in node_ids if n}
        with _LOCK:
            rows = [r for r in self._read(path)
                    if isinstance(r, str) or r.get("node_id") not in drop]
            self._write(path, rows)

    # ── queries ───────────────────────────────────────────────────

    def state(self, dataset: str) -> dict:
        """UI-facing state: node-keyed labels, pair ids, saved-single
        node ids + row count (no content)."""
        kto = {}
        for r in self._read(self._path(dataset, "kto")):
            if isinstance(r, dict) and r.get("node_id"):
                kto[r["node_id"]] = bool(r.get("label"))
        dpo = []
        for r in self._read(self._path(dataset, "dpo")):
            if not isinstance(r, dict):
                continue
            # Bulk rows may carry node ids on one side only (review-flow
            # saves) or none at all (unattended runs) — they still count.
            if r.get("chosen_node_id") or r.get("rejected_node_id") \
                    or r.get("source") == "bulk":
                dpo.append({
                    "chosen": r.get("chosen_node_id"),
                    "rejected": r.get("rejected_node_id"),
                    "source": r.get("source", "manual"),
                })
        # Single rows: node ids drive the 💾/✓ badge; the count includes
        # unattended rows (no node id) and any hand-added JSON rows.
        single_nodes, single_count = [], 0
        for r in self._read(self._path(dataset, "single")):
            if not isinstance(r, dict):
                continue
            single_count += 1
            if r.get("node_id"):
                single_nodes.append(r["node_id"])
        return {"kto": kto, "dpo": dpo,
                "single": {"nodes": single_nodes, "count": single_count}}

    def list_datasets(self) -> list[str]:
        if not self.root.is_dir():
            return []
        names = set()
        for p in self.root.iterdir():
            for kind in (".single.jsonl", ".kto.jsonl", ".dpo.jsonl"):
                if p.name.endswith(kind):
                    names.add(p.name[: -len(kind)])
        return sorted(names)
