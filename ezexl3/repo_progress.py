from __future__ import annotations

import re
import shutil
import sys
from typing import Dict


_ANSI_STRIP_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return _ANSI_STRIP_RE.sub("", text)


_BOX_DRAWING_RE = re.compile(r"[\u2500-\u257f]{10,}")


def _gpu_status_line(gpu_id: int, text: str, cols: int) -> str:
    """Build a single GPU status line, fitted to *cols* to prevent wrapping.

    If the text contains a progress bar (long run of box-drawing characters),
    the bar is shrunk proportionally so the label and time info at the ends
    are preserved. Falls back to right-truncation otherwise.
    """
    prefix = f"  GPU {gpu_id} | "
    max_text = cols - len(prefix) - 1
    if max_text <= 0 or len(text) <= max_text:
        return f"\033[2K{prefix}{text}"

    m = _BOX_DRAWING_RE.search(text)
    if m:
        bar = m.group()
        excess = len(text) - max_text
        new_len = max(4, len(bar) - excess)
        step = len(bar) / new_len
        shrunken = "".join(bar[int(i * step)] for i in range(new_len))
        text = text[: m.start()] + shrunken + text[m.end() :]

    if len(text) > max_text:
        text = text[: max_text - 1] + "…"
    return f"\033[2K{prefix}{text}"


def _clear_and_redraw_progress(gpu_status: Dict[int, str], num_lines: int) -> None:
    """Overwrite the last *num_lines* in-place with the current *gpu_status*."""
    cols = shutil.get_terminal_size((80, 24)).columns
    sys.stdout.write(f"\033[{num_lines}A")
    for gpu_id in sorted(gpu_status):
        sys.stdout.write(_gpu_status_line(gpu_id, gpu_status[gpu_id], cols) + "\n")
    sys.stdout.flush()


def _print_above_progress(message: str, gpu_status: Dict[int, str], num_lines: int) -> None:
    """Print *message* above the fixed progress area, then redraw it."""
    cols = shutil.get_terminal_size((80, 24)).columns
    sys.stdout.write(f"\033[{num_lines}A")
    for _ in range(num_lines):
        sys.stdout.write("\033[2K\n")
    sys.stdout.write(f"\033[{num_lines}A")
    sys.stdout.write(f"{message}\n")
    for gpu_id in sorted(gpu_status):
        sys.stdout.write(_gpu_status_line(gpu_id, gpu_status[gpu_id], cols) + "\n")
    sys.stdout.flush()


def _build_synthetic_bar(pct: int, width: int = 30) -> str:
    """Build a Unicode progress bar string from a percentage (0-100)."""
    pct = max(0, min(100, pct))
    filled = int(width * pct / 100)
    empty = width - filled
    return "\u2501" * filled + "\u2500" * empty + f" {pct:3d}%"


def _init_gpu_progress(use_ansi: bool, gpu_status: Dict[int, str]) -> None:
    """Print the initial GPU status area (one line per GPU)."""
    if use_ansi:
        for d in sorted(gpu_status):
            sys.stdout.write(f"\033[2K  GPU {d} | idle\n")
        sys.stdout.flush()


def _redraw_gpu_progress(use_ansi: bool, gpu_status: Dict[int, str], num_lines: int) -> None:
    """Update the GPU progress area in-place."""
    if use_ansi:
        _clear_and_redraw_progress(gpu_status, num_lines)
    else:
        for gpu_id in sorted(gpu_status):
            text = gpu_status[gpu_id]
            if text != "idle":
                sys.stdout.write(f"\rgpu{gpu_id}:  GPU {gpu_id} | {text}\n")
        sys.stdout.flush()


def _print_msg_with_progress(msg: str, use_ansi: bool, gpu_status: Dict[int, str], num_lines: int) -> None:
    """Print a message, preserving the GPU progress area if ANSI is available."""
    if use_ansi:
        _print_above_progress(msg, gpu_status, num_lines)
    else:
        print(msg)


def _cleanup_gpu_progress(use_ansi: bool, num_lines: int) -> None:
    """Clear the GPU progress area after all tasks are done."""
    if use_ansi:
        sys.stdout.write(f"\033[{num_lines}A")
        for _ in range(num_lines):
            sys.stdout.write("\033[2K\n")
        sys.stdout.write(f"\033[{num_lines}A")
        sys.stdout.flush()
