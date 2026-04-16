# ezexl3/evals.py
"""
Optional evaluation script integration for exllamav3 eval/ scripts.

Each eval is discovered from the exllamav3 install, run as a subprocess,
and its results are persisted to the measurement database.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, IO, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing import Queue

from ezexl3.measure_db import _EVAL_COL_TO_CSV, read_all_rows

# ---------------------------------------------------------------------------
# Vendored script paths
# ---------------------------------------------------------------------------

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")


def _vendor_script(name: str) -> str:
    """Return the path to a vendored eval script."""
    return os.path.join(_VENDOR_DIR, f"eval_{name}.py")


# ---------------------------------------------------------------------------
# Eval registry
# ---------------------------------------------------------------------------

@dataclass
class EvalDef:
    """Definition for one evaluation script."""
    name: str                              # e.g. "mmlu"
    script_name: str                       # e.g. "mmlu.py"
    cli_short: str                         # e.g. "-mmlu"
    cli_long: str                          # e.g. "--mmlu"
    db_columns: List[str]                  # DB column names this eval writes
    phase_label: str                       # short tag for progress display, e.g. "MMLU"
    needs_prompt_format: bool = False
    needs_all_gpus: bool = False           # run sequentially with all GPUs (perf, longctx)
    # For output files (humaneval, ifbench)
    output_subdir: Optional[str] = None    # e.g. "evals/humaneval"
    output_ext: str = ".jsonl"


EVAL_REGISTRY: Dict[str, EvalDef] = {}


def _register(e: EvalDef) -> EvalDef:
    EVAL_REGISTRY[e.name] = e
    return e


_register(EvalDef(
    name="diversity",
    script_name="diversity.py",
    cli_short="-div",
    cli_long="--diversity",
    db_columns=["diversity_score"],
    phase_label="DIV",
))

_register(EvalDef(
    name="humaneval",
    script_name="humaneval.py",
    cli_short="-he",
    cli_long="--humaneval",
    db_columns=["humaneval_pass"],
    phase_label="HEVAL",
    needs_prompt_format=True,
    output_subdir="evals/humaneval",
))

_register(EvalDef(
    name="ifbench",
    script_name="ifbench.py",
    cli_short="-ifb",
    cli_long="--ifbench",
    db_columns=["ifbench_score"],
    phase_label="IFB",
    needs_prompt_format=True,
    output_subdir="evals/ifbench",
))

_register(EvalDef(
    name="longctx",
    script_name="longctx.py",
    cli_short="-lctx",
    cli_long="--longctx",
    db_columns=["longctx_score"],
    phase_label="LCTX",
    needs_all_gpus=True,
))

_register(EvalDef(
    name="mmlu",
    script_name="mmlu.py",
    cli_short="-mmlu",
    cli_long="--mmlu",
    db_columns=["mmlu_accuracy"],
    phase_label="MMLU",
))

_register(EvalDef(
    name="perf",
    script_name="perf.py",
    cli_short="-perf",
    cli_long="--perf",
    db_columns=["perf_prefill_tps", "perf_gen_tps"],
    phase_label="PERF",
    needs_all_gpus=True,
))

# Ordered shortest-first for queue scheduling.
EVAL_QUEUE_ORDER = ["longctx", "diversity", "perf", "mmlu", "humaneval", "ifbench"]


# ---------------------------------------------------------------------------
# Script discovery (vendored)
# ---------------------------------------------------------------------------

def find_eval_script(eval_name: str) -> str:
    """Return the full path to the vendored eval script."""
    path = _vendor_script(eval_name)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Vendored eval script not found: {path}. "
            "This is a packaging error — the script should be bundled with ezexl3."
        )
    return path


# ---------------------------------------------------------------------------
# Prompt format detection
# ---------------------------------------------------------------------------

_FORMAT_KEYWORDS = {
    "llama": "Llama",
    "chatml": "ChatML",
    "mistral": "Mistral",
    "gemma": "Gemma",
    "qwen": "Qwen",
    "phi": "Phi",
    "command": "Command",
    "deepseek": "DeepSeek",
    "vicuna": "Vicuna",
}


def detect_prompt_format(model_dir: str) -> Optional[str]:
    """Try to auto-detect prompt format from tokenizer_config.json or model name."""
    config_path = os.path.join(model_dir, "tokenizer_config.json")
    model_name = os.path.basename(os.path.abspath(model_dir)).lower()

    # Check tokenizer_config.json for chat_template hints
    if os.path.isfile(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            template = str(config.get("chat_template", "")).lower()
            for keyword, fmt in _FORMAT_KEYWORDS.items():
                if keyword in template:
                    return fmt
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: match model directory name
    for keyword, fmt in _FORMAT_KEYWORDS.items():
        if keyword in model_name:
            return fmt

    return None


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def build_eval_cmd(
    eval_name: str,
    model_dir: str,
    device: int,
    base_dir: str,
    label: str,
    eval_arg: int | bool = 0,
    num_devices: int = 1,
) -> List[str]:
    """Build the subprocess command for an eval script.

    Note: device selection is handled via CUDA_VISIBLE_DEVICES in the
    subprocess environment, not via CLI args.  The eval scripts use
    exllamav3's ``model_init`` which accepts ``-gs`` (GPU split / VRAM
    allocation) but not ``-d``.

    When *num_devices* > 1 the command includes ``-gs 99,99,...`` so
    model_init splits across all visible GPUs.
    """
    # The perf eval is invoked through ezexl3.perf_runner, a wrapper that
    # monkey-patches heartbeat output into the (otherwise unmodified)
    # vendored script. Other evals run their vendored script directly.
    if eval_name == "perf":
        cmd = [sys.executable, "-m", "ezexl3.perf_runner", "-m", model_dir]
    else:
        script_path = find_eval_script(eval_name)
        # All eval scripts use exllamav3's model_init which takes -m for model dir.
        # Device is set via CUDA_VISIBLE_DEVICES (handled by the subprocess runner).
        cmd = [sys.executable, script_path, "-m", model_dir]

    # Multi-GPU: tell model_init to split across all visible devices.
    if num_devices > 1:
        cmd += ["-gs", ",".join("99" for _ in range(num_devices))]

    if eval_name == "diversity":
        n_samples = eval_arg if isinstance(eval_arg, int) and eval_arg > 0 else 50
        cmd += ["-samples", str(n_samples)]

    elif eval_name == "humaneval":
        spt = eval_arg if isinstance(eval_arg, int) and eval_arg > 0 else 200
        out_dir = os.path.join(base_dir, "evals", "humaneval")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{label}.jsonl")
        cmd += ["-o", out_file, "-spt", str(spt), "-e"]
        prompt_fmt = detect_prompt_format(model_dir)
        if prompt_fmt:
            cmd += ["-pf", prompt_fmt]

    elif eval_name == "ifbench":
        max_tokens = eval_arg if isinstance(eval_arg, int) and eval_arg > 0 else 16384
        out_dir = os.path.join(base_dir, "evals", "ifbench")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{label}.jsonl")
        cmd += ["-o", out_file, "-mt", str(max_tokens), "-e"]

    elif eval_name == "longctx":
        # longctx needs large cache for long document comprehension
        cmd += ["-cs", "65536"]

    elif eval_name == "mmlu":
        fewshot = eval_arg if isinstance(eval_arg, int) and eval_arg > 0 else 5
        cmd += ["-fs", str(fewshot)]

    elif eval_name == "perf":
        max_length = eval_arg if isinstance(eval_arg, int) and eval_arg > 0 else 32768
        cmd += ["-max_length", str(max_length)]

    return cmd


# ---------------------------------------------------------------------------
# Checkpoint: check if eval already has results
# ---------------------------------------------------------------------------

def eval_has_result(db_path: str, label: str, eval_name: str) -> bool:
    """Check if the DB already has non-empty values for this eval's columns."""
    eval_def = EVAL_REGISTRY[eval_name]
    rows = read_all_rows(db_path)
    row = rows.get(label, {})
    csv_cols = [_EVAL_COL_TO_CSV[c] for c in eval_def.db_columns]
    return all(bool((row.get(col) or "").strip()) for col in csv_cols)


def eval_has_output_file(base_dir: str, eval_name: str, label: str) -> bool:
    """For evals that produce output files (humaneval, ifbench), check existence."""
    eval_def = EVAL_REGISTRY[eval_name]
    if not eval_def.output_subdir:
        return False
    out_file = os.path.join(base_dir, eval_def.output_subdir, f"{label}{eval_def.output_ext}")
    return os.path.isfile(out_file)


# ---------------------------------------------------------------------------
# Progress parsers
# ---------------------------------------------------------------------------
# Each parser receives a single line (stripped of ANSI codes) and returns
# a short display string or None if the line is not a progress indicator.

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ProgressBar from rich/exllamav3 outputs like: "Description  42%"
# or lines with N/M patterns
_PCT_RE = re.compile(r"(\d+)%")
_FRACTION_RE = re.compile(r"(\d+)\s*/\s*(\d+)")


def _parse_diversity_progress(line: str) -> Optional[str]:
    """Parse diversity.py output for progress."""
    clean = _strip_ansi(line).strip()
    if not clean:
        return None
    # Generation phase progress bar
    m = _PCT_RE.search(clean)
    if m:
        pct = m.group(1)
        if "extract" in clean.lower() or "analyz" in clean.lower():
            return f"extract {pct}%"
        return f"gen {pct}%"
    # Mean diversity result line
    if clean.startswith("mean"):
        return "complete"
    return None


def _parse_humaneval_progress(line: str) -> Optional[str]:
    """Parse humaneval.py output for progress."""
    clean = _strip_ansi(line).strip()
    if not clean:
        return None
    # "** Problem X, sample Y / Z"
    m = re.search(r"Problem\s+(\d+),\s+sample\s+(\d+)\s*/\s*(\d+)", clean)
    if m:
        return f"p{m.group(1)} s{m.group(2)}/{m.group(3)}"
    # ProgressBar percentage
    m = _PCT_RE.search(clean)
    if m:
        pct = m.group(1)
        if "creat" in clean.lower():
            return f"setup {pct}%"
        return f"gen {pct}%"
    # Saving output
    if "Saving:" in clean or "saved" in clean.lower():
        return "saving"
    return None


def _parse_ifbench_progress(line: str) -> Optional[str]:
    """Parse ifbench.py output for progress."""
    clean = _strip_ansi(line).strip()
    if not clean:
        return None
    # "pending: N  active N  TPS tokens/s"
    m = re.search(r"pending:\s*(\d+)\s+active\s+(\d+)\s+([\d.]+)\s+tokens/s", clean)
    if m:
        return f"pend {m.group(1)} act {m.group(2)} {m.group(3)} t/s"
    # ProgressBar percentage
    m = _PCT_RE.search(clean)
    if m:
        return f"{m.group(1)}%"
    # Written output
    if "Responses written" in clean or "written to" in clean.lower():
        return "complete"
    return None


def _parse_longctx_progress(line: str) -> Optional[str]:
    """Parse longctx.py output for progress."""
    clean = _strip_ansi(line).strip()
    if not clean:
        return None
    # Test section headers
    m = re.search(r"([A-Z\s&]+TEST)", clean)
    if m:
        return m.group(1).strip().lower()
    # ProgressBar percentage
    m = _PCT_RE.search(clean)
    if m:
        return f"inference {m.group(1)}%"
    return None


def _parse_mmlu_progress(line: str) -> Optional[str]:
    """Parse mmlu.py output for progress."""
    clean = _strip_ansi(line).strip()
    if not clean:
        return None
    # Per-subject result line: "biology:   125/ 150 = 83.33% correct"
    m = re.search(r"^(\w[\w\s]*?):\s+\d+\s*/\s*\d+\s*=\s*([\d.]+)%", clean)
    if m:
        subj = m.group(1).strip()[:20]
        return f"{subj} {m.group(2)}%"
    # ProgressBar percentage
    m = _PCT_RE.search(clean)
    if m:
        pct = m.group(1)
        cl = clean.lower()
        if "preprompt" in cl:
            return f"preprompts {pct}%"
        elif "question" in cl:
            return f"questions {pct}%"
        elif "test" in cl:
            return f"testing {pct}%"
        return f"{pct}%"
    return None


def _parse_perf_progress(line: str) -> Optional[str]:
    """Parse perf.py output for progress."""
    clean = _strip_ansi(line).strip()
    if not clean:
        return None
    # Inner-loop heartbeat (emitted by vendored eval_perf.py)
    if clean.startswith("PERF_HEARTBEAT"):
        # Strip the marker prefix and pass the rest through for display.
        return clean[len("PERF_HEARTBEAT"):].strip()
    # Throughput result lines
    m = re.search(r"(Length|Context)\s+(\d+):\s+([\d.]+)\s+tokens/s", clean)
    if m:
        kind = "prefill" if m.group(1) == "Length" else "gen"
        return f"{kind} @{m.group(2)}: {m.group(3)} t/s"
    # Section headers
    if "prefill" in clean.lower() and len(clean) < 30:
        return "prefill phase"
    if "generation" in clean.lower() and len(clean) < 30:
        return "generation phase"
    # ProgressBar percentage
    m = _PCT_RE.search(clean)
    if m:
        pct = m.group(1)
        if "warmup" in clean.lower():
            return f"warmup {pct}%"
        return f"measuring {pct}%"
    return None


# Map eval name -> progress parser function
PROGRESS_PARSERS: Dict[str, Callable[[str], Optional[str]]] = {
    "diversity": _parse_diversity_progress,
    "humaneval": _parse_humaneval_progress,
    "ifbench": _parse_ifbench_progress,
    "longctx": _parse_longctx_progress,
    "mmlu": _parse_mmlu_progress,
    "perf": _parse_perf_progress,
}


# ---------------------------------------------------------------------------
# Result extractors
# ---------------------------------------------------------------------------
# Each extractor receives the full captured stdout and returns a dict
# mapping DB column names -> string values suitable for upsert_row().

_DIVERSITY_MEAN_RE = re.compile(r"^mean\s+([\d.]+)", re.MULTILINE)

_HUMANEVAL_PASS_RE = re.compile(
    r"pass@1.*?:\s*([\d.]+)", re.IGNORECASE
)

_IFBENCH_SCORE_RE = re.compile(
    r"(?:score|accuracy|correct).*?:\s*([\d.]+)", re.IGNORECASE
)

_LONGCTX_TESTS = [
    "SUMMARY TEST", "FRENCH TEST", "ZOOMER TEST",
    "Q&A TEST", "CORRUPTION TEST", "NAME EXTRACTION TEST",
]

_MMLU_ACCURACY_RE = re.compile(
    r"all\s+subjects.*?=\s*([\d.]+)%", re.IGNORECASE
)

_PERF_PREFILL_RE = re.compile(
    r"Length\s+(\d+):\s+([\d.]+)\s+tokens/s"
)
_PERF_GEN_RE = re.compile(
    r"Context\s+(\d+):\s+([\d.]+)\s+tokens/s"
)


def _extract_diversity_result(output: str) -> dict:
    clean = _strip_ansi(output)
    m = _DIVERSITY_MEAN_RE.search(clean)
    if m:
        return {"diversity_score": m.group(1)}
    return {"diversity_score": ""}


def _extract_humaneval_result(output: str) -> dict:
    clean = _strip_ansi(output)
    m = _HUMANEVAL_PASS_RE.search(clean)
    if m:
        return {"humaneval_pass": m.group(1)}
    # Fallback: count completions
    saving = re.search(r"Saving:", clean)
    if saving:
        return {"humaneval_pass": "done"}
    return {"humaneval_pass": ""}


def _extract_ifbench_result(output: str) -> dict:
    clean = _strip_ansi(output)
    m = _IFBENCH_SCORE_RE.search(clean)
    if m:
        return {"ifbench_score": m.group(1)}
    if "Responses written" in clean:
        return {"ifbench_score": "done"}
    return {"ifbench_score": ""}


def _extract_longctx_result(output: str) -> dict:
    """Count how many test sections appear in the output."""
    clean = _strip_ansi(output)
    found = sum(1 for t in _LONGCTX_TESTS if t in clean)
    total = len(_LONGCTX_TESTS)
    if found > 0:
        return {"longctx_score": f"{found}/{total}"}
    return {"longctx_score": ""}


def _extract_mmlu_result(output: str) -> dict:
    clean = _strip_ansi(output)
    m = _MMLU_ACCURACY_RE.search(clean)
    if m:
        return {"mmlu_accuracy": f"{m.group(1)}%"}
    return {"mmlu_accuracy": ""}


def _extract_perf_result(output: str) -> dict:
    clean = _strip_ansi(output)
    # Take the last reported prefill and generation throughput values
    prefill_matches = _PERF_PREFILL_RE.findall(clean)
    gen_matches = _PERF_GEN_RE.findall(clean)

    result: dict = {}
    if prefill_matches:
        # Use the value at the longest tested length
        result["perf_prefill_tps"] = prefill_matches[-1][1]
    else:
        result["perf_prefill_tps"] = ""

    if gen_matches:
        # Use the value at context length 0 (pure generation speed)
        result["perf_gen_tps"] = gen_matches[0][1]
    else:
        result["perf_gen_tps"] = ""

    return result


def extract_perf_detail(output: str) -> Dict[str, List[tuple]]:
    """Extract all prefill/generation rows from perf.py output.

    Returns ``{"prefill": [(length, tps), ...], "generation": [(ctx, tps), ...]}``.
    Each tuple is ``(context_length_int, tokens_per_second_float)``.
    """
    clean = _strip_ansi(output)
    prefill = [
        (int(m[0]), float(m[1]))
        for m in _PERF_PREFILL_RE.findall(clean)
    ]
    generation = [
        (int(m[0]), float(m[1]))
        for m in _PERF_GEN_RE.findall(clean)
    ]
    return {"prefill": prefill, "generation": generation}


RESULT_EXTRACTORS: Dict[str, Callable[[str], dict]] = {
    "diversity": _extract_diversity_result,
    "humaneval": _extract_humaneval_result,
    "ifbench": _extract_ifbench_result,
    "longctx": _extract_longctx_result,
    "mmlu": _extract_mmlu_result,
    "perf": _extract_perf_result,
}


# ---------------------------------------------------------------------------
# Result display formatters
# ---------------------------------------------------------------------------
# Formats an eval's result_dict (the dict returned by RESULT_EXTRACTORS) into
# a short human-readable string for the "DONE" progress line. Keeps the
# presentation next to the regexes so adding a new eval only requires touching
# one file.


def _fmt(val: str) -> str:
    """Empty string → 'N/A', otherwise return value unchanged."""
    return val if val else "N/A"


def format_eval_result(eval_name: str, result_dict: Dict[str, str]) -> str:
    """Render an eval's result dict for display in the 'DONE' message.

    Takes DB-column-keyed values (e.g. ``perf_prefill_tps``) and returns a
    compact, human-readable summary. Returns an empty string for unknown
    evals.
    """
    if eval_name == "diversity":
        return f"diversity={_fmt(result_dict.get('diversity_score', ''))}"
    if eval_name == "humaneval":
        return f"pass@1={_fmt(result_dict.get('humaneval_pass', ''))}"
    if eval_name == "ifbench":
        return f"score={_fmt(result_dict.get('ifbench_score', ''))}"
    if eval_name == "longctx":
        return f"tests={_fmt(result_dict.get('longctx_score', ''))}"
    if eval_name == "mmlu":
        return f"accuracy={_fmt(result_dict.get('mmlu_accuracy', ''))}"
    if eval_name == "perf":
        prefill = _fmt(result_dict.get("perf_prefill_tps", ""))
        gen = _fmt(result_dict.get("perf_gen_tps", ""))
        return f"prefill={prefill} t/s, gen={gen} t/s"
    return ""


def result_is_empty(eval_name: str, result_dict: Dict[str, str]) -> bool:
    """Return True if the extractor produced no usable values.

    Used by the measure loop to warn when an eval ran to completion but the
    extractor regex didn't match anything, which otherwise looks identical to
    a successful run.
    """
    eval_def = EVAL_REGISTRY.get(eval_name)
    if eval_def is None:
        return True
    return not any((result_dict.get(col) or "").strip() for col in eval_def.db_columns)


# ---------------------------------------------------------------------------
# Generic eval subprocess runner
# ---------------------------------------------------------------------------

def run_eval_subprocess(
    cmd: List[str],
    device: int,
    results: "Queue[Optional[dict]]",
    phase_label: str,
    eval_name: str,
    log_f: Optional[IO] = None,
    cuda_visible_devices: Optional[str] = None,
) -> str:
    """Run an eval subprocess with progress parsing and output capture.

    Handles both \\n and \\r delimited lines (for progress bar overwrites).
    Falls back to asymptotic ticker if no progress detected.
    Returns the full captured output.
    """
    if log_f:
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=0, env=env,
    )
    assert proc.stdout is not None

    progress_parser = PROGRESS_PARSERS.get(eval_name)
    buf: List[bytes] = []
    last_send: float = 0.0
    last_progress_time: float = time.monotonic()
    had_progress = False
    load_start = time.monotonic()

    # Per-eval updating heartbeat in the parent's stdout. The leading
    # "\r{eval_name}:" prefix lets the UI's terminal.js group these into
    # a single line that updates in place (instead of cascading into
    # scrollback). Throttled separately from the in-place progress event.
    last_stdout_hb: float = 0.0
    stdout_hb_interval: float = 1.5
    stdout_progress_key = f"{eval_name}:"

    def _emit_stdout_progress(text: str) -> None:
        sys.stdout.write(f"\r{stdout_progress_key} {phase_label} | {text}\n")
        sys.stdout.flush()

    def _emit_stdout_log(text: str) -> None:
        # Permanent (non-overwriting) line — used for completed-length
        # result rows that should stay visible in the UI scrollback.
        sys.stdout.write(f"[{eval_name}] {phase_label} | {text}\n")
        sys.stdout.flush()

    # Asymptotic ticker for loading phase (same pattern as catbench)
    ticker_stop = threading.Event()

    def _asymptotic_ticker() -> None:
        while not ticker_stop.wait(timeout=0.5):
            if had_progress:
                ticker_stop.set()
                return
            elapsed = time.monotonic() - load_start
            pct = int(95 * (1 - 2 ** (-elapsed / 30)))
            results.put({
                "event": "progress", "device": device,
                "text": f"{phase_label} | loading {pct}%",
            })

    ticker_thread = threading.Thread(target=_asymptotic_ticker, daemon=True)
    ticker_thread.start()

    # Read byte-by-byte to handle \r overwrites
    line_buf = bytearray()
    while True:
        chunk = proc.stdout.read(4096)
        if not chunk:
            break
        buf.append(chunk)
        line_buf.extend(chunk)

        # Split on both \r and \n to get progress lines
        while b"\n" in line_buf or b"\r" in line_buf:
            # Find the earliest delimiter
            idx_n = line_buf.find(b"\n")
            idx_r = line_buf.find(b"\r")
            if idx_n == -1:
                idx = idx_r
            elif idx_r == -1:
                idx = idx_n
            else:
                idx = min(idx_n, idx_r)

            line_bytes = bytes(line_buf[:idx])
            line_buf = line_buf[idx + 1:]

            try:
                line_text = line_bytes.decode("utf-8", errors="replace")
            except Exception:
                continue

            if log_f:
                log_f.write(line_text + "\n")

            if not progress_parser:
                continue

            progress_text = progress_parser(line_text)
            if progress_text is not None:
                if not had_progress:
                    had_progress = True
                    ticker_stop.set()
                now = time.monotonic()
                if now - last_send >= 0.5:
                    results.put({
                        "event": "progress", "device": device,
                        "text": f"{phase_label} | {progress_text}",
                    })
                    last_send = now
                # Per-length perf result lines ("prefill @256: 1000.00 t/s",
                # "gen @512: 110.00 t/s") get a permanent log entry; everything
                # else (PERF_HEARTBEAT, ProgressBar %, section headers) goes to
                # the throttled in-place updating line.
                is_perf_result = (
                    eval_name == "perf"
                    and (progress_text.startswith("prefill @")
                         or progress_text.startswith("gen @"))
                )
                if is_perf_result:
                    _emit_stdout_log(progress_text)
                    last_stdout_hb = now
                elif now - last_stdout_hb >= stdout_hb_interval:
                    _emit_stdout_progress(progress_text)
                    last_stdout_hb = now
                last_progress_time = now

    # Process any remaining bytes in line_buf
    if line_buf:
        line_text = bytes(line_buf).decode("utf-8", errors="replace")
        if log_f:
            log_f.write(line_text + "\n")
        if progress_parser:
            progress_text = progress_parser(line_text)
            if progress_text is not None:
                results.put({
                    "event": "progress", "device": device,
                    "text": f"{phase_label} | {progress_text}",
                })

    ticker_stop.set()
    ticker_thread.join(timeout=2)

    proc.wait()
    full_output = b"".join(buf).decode("utf-8", errors="replace")

    if log_f:
        log_f.flush()

    if proc.returncode != 0:
        # Non-zero exit. If the subprocess printed usable output before
        # crashing (e.g. CUDA OOM/illegal-memory-access on the final
        # context length after most results were already reported),
        # surface a warning and return what we captured so downstream
        # extractors can salvage partial results. If no parseable output
        # exists, fall through and raise — the caller will report it as
        # a full failure.
        tail = full_output[-2000:]
        warn = (
            f"⚠️  Eval {eval_name} exited with code {proc.returncode} "
            f"(attempting to salvage partial output)"
        )
        if log_f:
            log_f.write(warn + "\n")
            log_f.flush()
        parser = RESULT_EXTRACTORS.get(eval_name)
        partial_ok = False
        if parser is not None:
            try:
                partial_result = parser(full_output)
                partial_ok = any(
                    (v or "").strip() for v in partial_result.values()
                )
            except Exception:
                partial_ok = False
        if partial_ok:
            results.put({
                "event": "progress", "device": device,
                "text": f"{phase_label} | exited code {proc.returncode}, salvaged partial",
            })
            # Also surface a permanent log line so the salvage is visible
            # in the UI terminal scrollback (the progress event above gets
            # overwritten by subsequent updates / the final DONE line).
            print(
                f"⚠️  [{eval_name}] {phase_label} crashed with exit code "
                f"{proc.returncode} — partial results captured (see {phase_label}'s DONE summary below)"
            )
            sys.stdout.flush()
            return full_output
        raise RuntimeError(
            f"Eval {eval_name} failed with exit code {proc.returncode}: "
            f"{' '.join(cmd)}\n\nOutput:\n{tail}"
        )

    return full_output
