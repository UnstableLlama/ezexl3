# Prompt queue for DPO preference capture: feed a JSONL file of prompts
# through the chat UI's duel flow one line at a time.
#
# Each line yields one user prompt. Accepted line shapes:
#   "raw text"                          — a JSON string
#   {"prompt": "..."}                   — also: text / instruction / question
#   {"prompt": [{role, content}, ...]}  — turn list; the last user turn's
#   [{role, content}, ...]                content is used
#   any non-JSON line                   — used verbatim (plain-text lists)
#
# Progress is checkpointed per queue file in <ratings_dir>/
# queue_checkpoints.json as the next unserved 1-based FILE line number, so
# a queue re-opened after a browser or server restart resumes where it
# left off. An explicit start line overrides the checkpoint.

from __future__ import annotations

import json
import threading
from pathlib import Path

# Object keys probed (in order) for the prompt text of a JSON-object line.
_PROMPT_KEYS = ("prompt", "text", "instruction", "question", "message")

CHECKPOINT_FILE = "queue_checkpoints.json"

# Serializes checkpoint read-modify-write across aiohttp's to_thread workers.
_LOCK = threading.Lock()


def _from_turns(turns) -> str | None:
    """Content of the last user turn in a {role, content} list, or None."""
    text = None
    for turn in turns:
        if (isinstance(turn, dict) and turn.get("role") == "user"
                and isinstance(turn.get("content"), str)):
            text = turn["content"]
    return text if text and text.strip() else None


def parse_prompt_line(raw: str) -> str | None:
    """One JSONL line -> prompt text; None for blank lines.

    Raises ValueError for lines that parse as JSON but hold no usable
    prompt (the caller adds the line number).
    """
    s = raw.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        return s  # plain-text prompt list — use the line verbatim
    if isinstance(obj, str):
        if obj.strip():
            return obj
        raise ValueError("empty prompt string")
    if isinstance(obj, dict):
        for key in _PROMPT_KEYS:
            v = obj.get(key)
            if isinstance(v, str) and v.strip():
                return v
            if key == "prompt" and isinstance(v, list):
                text = _from_turns(v)
                if text:
                    return text
        raise ValueError(
            f"no prompt found (looked for keys: {', '.join(_PROMPT_KEYS)})")
    if isinstance(obj, list):
        text = _from_turns(obj)
        if text:
            return text
        raise ValueError("turn list has no user turn with string content")
    raise ValueError(f"cannot extract a prompt from {type(obj).__name__}")


class PromptQueue:
    """An open prompt file plus a cursor over its usable lines.

    Entries keep their 1-based file line numbers — blank lines are
    skipped but never renumber anything, so "start at line N" and the
    checkpoint always mean the line you'd see in an editor.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.entries: list[dict] = []  # {"line": int, "text": str}
        self.pos = 0                   # index of the next unserved entry
        text = self.path.read_text("utf-8")
        for line_no, raw in enumerate(text.splitlines(), start=1):
            try:
                prompt = parse_prompt_line(raw)
            except ValueError as e:
                raise ValueError(f"{self.path.name} line {line_no}: {e}")
            if prompt is not None:
                self.entries.append({"line": line_no, "text": prompt})
        if not self.entries:
            raise ValueError(f"{self.path.name}: no prompts found")

    def seek_line(self, line: int) -> None:
        """Move the cursor to the first entry at or after file line *line*."""
        self.pos = len(self.entries)
        for i, entry in enumerate(self.entries):
            if entry["line"] >= line:
                self.pos = i
                break

    def advance(self, index: int) -> bool:
        """Advance past entry *index* if it is the current one.

        The index guard makes advancing idempotent — a retried or
        duplicate request can't skip a prompt.
        """
        if index != self.pos or self.pos >= len(self.entries):
            return False
        self.pos += 1
        return True

    def next_line(self) -> int:
        """The checkpoint value: next unserved file line (or EOF + 1)."""
        if self.pos < len(self.entries):
            return self.entries[self.pos]["line"]
        return self.entries[-1]["line"] + 1

    def status(self) -> dict:
        cur = self.entries[self.pos] if self.pos < len(self.entries) else None
        return {
            "active": True,
            "path": str(self.path),
            "total": len(self.entries),
            "index": self.pos,
            "done": cur is None,
            "line": cur["line"] if cur else None,
            "prompt": cur["text"] if cur else None,
            "remaining": len(self.entries) - self.pos,
        }


# ── Checkpoints ──────────────────────────────────────────────────────

def _checkpoints_path(root: str | Path) -> Path:
    return Path(root).expanduser() / CHECKPOINT_FILE


def _queue_key(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def load_checkpoint(root: str | Path, path: str | Path) -> int | None:
    """Next unserved file line recorded for *path*, or None."""
    p = _checkpoints_path(root)
    with _LOCK:
        try:
            data = json.loads(p.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    entry = data.get(_queue_key(path)) if isinstance(data, dict) else None
    line = entry.get("line") if isinstance(entry, dict) else None
    return line if isinstance(line, int) and line >= 1 else None


def save_checkpoint(root: str | Path, path: str | Path, line: int) -> None:
    p = _checkpoints_path(root)
    with _LOCK:
        try:
            data = json.loads(p.read_text("utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, json.JSONDecodeError):
            data = {}
        data[_queue_key(path)] = {"line": line}
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), "utf-8")
        tmp.replace(p)
