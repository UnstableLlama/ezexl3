# ezexl3/repo.py
from __future__ import annotations

import csv
import importlib.util
import math
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from typing import Dict, IO, List, Optional, Tuple

from ezexl3.quantize import run as quant_run
from ezexl3.measure import (
    append_csv_row,
    default_csv_path,
    ensure_csv_exists,
    file_size_gib,
    find_model_diff_script,
    read_existing_field_labels,
    read_existing_weights,
)


def _normalize_bpw_str(raw: str) -> str:
    token = str(raw).strip()
    if not token:
        raise ValueError("Empty BPW value provided")
    try:
        numeric = float(token)
    except ValueError as e:
        raise ValueError(f"Invalid BPW value '{raw}'") from e
    if numeric <= 0:
        raise ValueError(f"BPW values must be > 0, got '{raw}'")

    if "." not in token:
        return str(int(numeric)) if numeric.is_integer() else token

    trimmed = token.rstrip("0").rstrip(".")
    if not trimmed:
        return str(int(numeric)) if numeric.is_integer() else token
    if "." not in trimmed and numeric.is_integer():
        return str(int(numeric))
    return trimmed


def _split_integer_optimized_bpws(bpws: List[str]) -> Tuple[List[str], List[str]]:
    integer_bpws: List[str] = []
    optimized_bpws: List[str] = []

    for raw in bpws:
        normalized = _normalize_bpw_str(raw)
        value = float(normalized)
        if math.isclose(value, round(value), abs_tol=1e-9):
            integer_bpws.append(str(int(round(value))))
        else:
            optimized_bpws.append(normalized)
    return integer_bpws, optimized_bpws


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _plan_repo_bpws(bpws: List[str]) -> Dict[str, List[str]]:
    ints, fracs = _split_integer_optimized_bpws(bpws)
    required_neighbors: List[str] = []
    for frac in fracs:
        frac_val = float(frac)
        low = math.floor(frac_val)
        high = math.ceil(frac_val)
        required_neighbors.extend([str(low), str(high)])

    requested_ints = _dedupe_preserve_order(ints)
    requested_fracs = _dedupe_preserve_order(fracs)
    quant_ints = _dedupe_preserve_order(requested_ints + required_neighbors)
    measure_targets = _dedupe_preserve_order(quant_ints + requested_fracs)

    return {
        "requested_integers": requested_ints,
        "requested_optimizeds": requested_fracs,
        "quant_integer_queue": quant_ints,
        "measure_queue": measure_targets,
    }


def _catbench_file_prefix(label: str) -> str:
    """Convert a CSV label to a catbench SVG filename prefix."""
    if label in ("bf16", "base"):
        return "bf16"
    try:
        val = float(label)
        return f"{val:.2f}bpw"
    except (ValueError, TypeError):
        return label


def _catbench_has_output(catbench_dir: str, file_prefix: str) -> bool:
    """Check if catbench already produced output for *file_prefix*.

    Returns True if any SVG output exists (canonical or numbered).
    Attempts re-extraction from .txt files first. Only SVGs count as
    output — unextractable .txt files do NOT prevent re-running inference.

    File naming convention:
      sample 1: {prefix}.svg / {prefix}.txt  (canonical)
      sample 2: {prefix}_1.svg / {prefix}_1.txt
      sample 3: {prefix}_2.svg / {prefix}_2.txt
    """
    if not os.path.isdir(catbench_dir):
        return False
    canonical_svg = os.path.join(catbench_dir, f"{file_prefix}.svg")
    if os.path.exists(canonical_svg):
        return True
    # Try re-extracting canonical .txt with latest extraction logic
    from ezexl3.catbench import extract_svg
    canonical_txt = os.path.join(catbench_dir, f"{file_prefix}.txt")
    if os.path.exists(canonical_txt):
        with open(canonical_txt, "r") as f:
            raw = f.read()
        svg_content = extract_svg(raw)
        if svg_content:
            with open(canonical_svg, "w") as f:
                f.write(svg_content)
            os.remove(canonical_txt)
            print(f"  🔄 Re-extracted SVG from {file_prefix}.txt ({len(svg_content)} chars)")
            return True
    # Check for numbered SVGs (_1.svg, _2.svg, ...)
    for fn in sorted(os.listdir(catbench_dir)):
        if fn.startswith(f"{file_prefix}_") and fn.endswith(".svg"):
            return True
    return False


def _resolve_exllamav3_util_scripts() -> Tuple[str, str]:
    attempted: List[str] = []
    roots: List[str] = []

    env_root = os.environ.get("EXLLAMAV3_ROOT", "").strip()
    if env_root:
        roots.append(env_root)

    spec = importlib.util.find_spec("exllamav3")
    if spec and spec.origin:
        pkg_dir = os.path.dirname(os.path.abspath(spec.origin))
        roots.extend(
            [
                os.path.dirname(pkg_dir),
                pkg_dir,
                os.path.join(pkg_dir, ".."),
            ]
        )

    checked_roots = []
    for root in roots:
        root_abs = os.path.abspath(root)
        if root_abs in checked_roots:
            continue
        checked_roots.append(root_abs)

        measure_path = os.path.join(root_abs, "util", "measure.py")
        optimize_path = os.path.join(root_abs, "util", "optimize.py")
        attempted.append(f"{measure_path} | {optimize_path}")
        if os.path.isfile(measure_path) and os.path.isfile(optimize_path):
            return measure_path, optimize_path

    attempted_msg = "\n  - ".join(attempted) if attempted else "(no paths discovered)"
    raise RuntimeError(
        "Could not locate exllamav3 util scripts measure.py and optimize.py. "
        "Set EXLLAMAV3_ROOT to your exllamav3 checkout root.\n"
        f"Attempted:\n  - {attempted_msg}"
    )


def _run_cmd(cmd: List[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


# ---------------------------------------------------------------------------
# Progress capture helpers
# ---------------------------------------------------------------------------

_ANSI_STRIP_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape sequences from *text*."""
    return _ANSI_STRIP_RE.sub("", text)


def _run_cmd_with_progress(
    cmd: List[str],
    device: int,
    results: "Queue[Optional[dict]]",
    log_f: Optional[IO] = None,
) -> str:
    """Run *cmd* in a PTY, stream output to *log_f*, and send throttled
    ``{"event": "progress", "device": …, "text": …}`` dicts through *results*.

    Falls back to a plain pipe if the PTY cannot be created.

    Returns the full captured output as a string.
    """
    if log_f:
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.flush()

    # --- try PTY first so the child thinks it has a real terminal ----------
    master_fd: Optional[int] = None
    try:
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            cmd, stdout=slave_fd, stderr=slave_fd, close_fds=True,
        )
        os.close(slave_fd)
    except Exception:
        # PTY unavailable – fall back to a pipe
        if master_fd is not None:
            os.close(master_fd)
            master_fd = None
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    last_send: float = 0.0
    buf = ""

    def _drain_fd(fd: int) -> bool:
        """Read available data from *fd*.  Returns False on EOF / error."""
        nonlocal buf, last_send
        try:
            data = os.read(fd, 4096)
        except OSError:
            return False
        if not data:
            return False
        text = data.decode("utf-8", errors="replace")
        if log_f:
            log_f.write(text)
            log_f.flush()
        buf += text
        _maybe_send_progress()
        return True

    def _drain_pipe() -> bool:
        """Read a line from the pipe stdout. Returns False on EOF."""
        nonlocal buf, last_send
        assert proc.stdout is not None
        line = proc.stdout.readline()
        if not line:
            return False
        text = line.decode("utf-8", errors="replace") if isinstance(line, bytes) else line
        if log_f:
            log_f.write(text)
            log_f.flush()
        buf += text
        _maybe_send_progress()
        return True

    def _maybe_send_progress() -> None:
        nonlocal buf, last_send
        now = time.monotonic()
        if now - last_send < 0.5:
            return
        # Extract the latest progress-bar segment: the last \r-separated
        # piece that has not been terminated by \n.
        lines = buf.split("\n")
        tail = lines[-1]  # incomplete line (no trailing \n)
        if "\r" in tail:
            segments = tail.split("\r")
            candidate = segments[-1].strip()
        else:
            candidate = tail.strip()
        if candidate:
            results.put({"event": "progress", "device": device, "text": _strip_ansi(candidate)})
            last_send = now

    # --- read loop ---------------------------------------------------------
    if master_fd is not None:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.5)
            if ready:
                if not _drain_fd(master_fd):
                    break
            elif proc.poll() is not None:
                # Process exited – drain anything left
                while _drain_fd(master_fd):
                    pass
                break
        os.close(master_fd)
    else:
        # Pipe fallback
        assert proc.stdout is not None
        while _drain_pipe():
            pass

    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}"
        )
    return buf


# ---------------------------------------------------------------------------
# ANSI progress-area rendering
# ---------------------------------------------------------------------------

_BOX_DRAWING_RE = re.compile(r"[\u2500-\u257f]{10,}")


def _gpu_status_line(gpu_id: int, text: str, cols: int) -> str:
    """Build a single GPU status line, fitted to *cols* to prevent wrapping.

    If the text contains a progress bar (long run of box-drawing characters),
    the bar is shrunk proportionally so the label and time info at the ends
    are preserved.  Falls back to right-truncation otherwise.
    """
    prefix = f"  GPU {gpu_id} | "
    max_text = cols - len(prefix) - 1  # -1 for safety margin
    if max_text <= 0 or len(text) <= max_text:
        return f"\033[2K{prefix}{text}"

    # Try to shrink a progress bar rather than chopping the tail
    m = _BOX_DRAWING_RE.search(text)
    if m:
        bar = m.group()
        excess = len(text) - max_text
        new_len = max(4, len(bar) - excess)
        step = len(bar) / new_len
        shrunken = "".join(bar[int(i * step)] for i in range(new_len))
        text = text[: m.start()] + shrunken + text[m.end() :]

    # Final safety clamp
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


def _print_above_progress(
    message: str,
    gpu_status: Dict[int, str],
    num_lines: int,
) -> None:
    """Print *message* above the fixed progress area, then redraw it."""
    cols = shutil.get_terminal_size((80, 24)).columns
    # Move up into the progress area and clear it
    sys.stdout.write(f"\033[{num_lines}A")
    for _ in range(num_lines):
        sys.stdout.write("\033[2K\n")
    # Move back up
    sys.stdout.write(f"\033[{num_lines}A")
    # Print the message (scrolls the terminal)
    sys.stdout.write(f"{message}\n")
    # Redraw the progress area
    for gpu_id in sorted(gpu_status):
        sys.stdout.write(_gpu_status_line(gpu_id, gpu_status[gpu_id], cols) + "\n")
    sys.stdout.flush()


def _build_synthetic_bar(pct: int, width: int = 30) -> str:
    """Build a Unicode progress bar string from a percentage (0-100).

    Uses box-drawing characters so the existing ``_gpu_status_line`` shrink
    logic can resize the bar proportionally when the terminal is narrow.
    """
    pct = max(0, min(100, pct))
    filled = int(width * pct / 100)
    empty = width - filled
    return "\u2501" * filled + "\u2500" * empty + f" {pct:3d}%"


def _build_optimized_jobs(model_dir: str, optimized_bpws: List[str]) -> Tuple[List[dict], List[dict]]:
    measurements_dir = os.path.join(model_dir, "measurements")
    os.makedirs(measurements_dir, exist_ok=True)

    compare_jobs_by_pair: Dict[Tuple[str, str], dict] = {}
    optimize_jobs: List[dict] = []

    for frac in optimized_bpws:
        frac_value = float(frac)
        low = str(math.floor(frac_value))
        high = str(math.ceil(frac_value))
        low_dir = os.path.join(model_dir, low)
        high_dir = os.path.join(model_dir, high)
        out_dir = os.path.join(model_dir, frac)
        measure_json = os.path.join(measurements_dir, f"{low}-{high}_measurement.json")

        if not os.path.isdir(low_dir):
            raise FileNotFoundError(f"Required lower integer quant not found for {frac}: {low_dir}")
        if not os.path.isdir(high_dir):
            raise FileNotFoundError(f"Required upper integer quant not found for {frac}: {high_dir}")

        compare_jobs_by_pair.setdefault(
            (low, high),
            {
                "low": low,
                "high": high,
                "low_dir": low_dir,
                "high_dir": high_dir,
                "measure_json": measure_json,
                "targets": [],
            },
        )
        compare_jobs_by_pair[(low, high)]["targets"].append(frac)

        optimize_jobs.append(
            {
                "optimized": frac,
                "out_dir": out_dir,
                "measure_json": measure_json,
                "low": low,
                "high": high,
            }
        )

    return list(compare_jobs_by_pair.values()), optimize_jobs


def _worker_optimized_compare(
    measure_script: str,
    model_dir: str,
    device: int,
    layers: int,
    tasks: "Queue[Optional[dict]]",
    results: "Queue[Optional[dict]]",
    log_path: Optional[str],
) -> None:
    import traceback

    log_f: Optional[IO] = None
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_f = open(log_path, "w")

    while True:
        job = tasks.get()
        if job is None:
            results.put(None)
            break

        label = f"{job['low']}-{job['high']}"
        results.put({"event": "start", "device": device, "job": job})
        try:
            cmd = [
                sys.executable,
                measure_script,
                "-i",
                job["low_dir"],
                job["high_dir"],
                "-r",
                model_dir,
                "-o",
                job["measure_json"],
                "-d",
                str(device),
                "-l",
                str(layers),
            ]
            _run_cmd_with_progress(cmd, device, results, log_f)
            results.put({"event": "done", "device": device, "job": job, "label": label})
        except Exception as e:
            if log_f:
                traceback.print_exc(file=log_f)
            results.put({"event": "error", "device": device, "job": job, "label": label, "error": str(e)})

    if log_f:
        log_f.flush()
        log_f.close()


def _run_optimized_compare_queue(
    model_dir: str,
    compare_jobs: List[dict],
    devices: List[int],
    measure_script: str,
    layers: int,
    write_logs: bool = True,
) -> None:
    if not compare_jobs:
        return
    if not devices:
        raise ValueError("No CUDA devices available for optimized comparative measure stage")

    tasks: Queue = Queue()
    results: Queue = Queue()

    for job in compare_jobs:
        tasks.put(job)
    for _ in devices:
        tasks.put(None)

    procs: List[Process] = []
    for device in devices:
        log_path = os.path.join(model_dir, "logs", f"optimized_compare_gpu{device}.log") if write_logs else None
        p = Process(
            target=_worker_optimized_compare,
            args=(measure_script, model_dir, device, layers, tasks, results, log_path),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    print(f"\n🚀 Optimized comparative measure: {len(compare_jobs)} jobs on {len(devices)} GPUs...")

    use_ansi = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    gpu_status: Dict[int, str] = {d: "idle" for d in devices}
    num_lines = len(devices)

    # Print initial progress area (one line per GPU)
    if use_ansi:
        for d in sorted(gpu_status):
            sys.stdout.write(f"\033[2K  GPU {d} | idle\n")
        sys.stdout.flush()

    active_workers = len(devices)
    failures = 0
    while active_workers > 0:
        res = results.get()
        if res is None:
            active_workers -= 1
            continue
        gpu = res["device"]
        event = res["event"]

        if event == "progress":
            gpu_status[gpu] = res["text"]
            if use_ansi:
                _clear_and_redraw_progress(gpu_status, num_lines)
            continue

        job = res["job"]
        label = f"{job['low']}-{job['high']}"
        targets = ",".join(job["targets"])

        if event == "start":
            msg = f"🧪 [GPU {gpu}] START compare {label} for target(s): {targets}"
            gpu_status[gpu] = f"{label} | starting..."
        elif event == "done":
            msg = f"✅ [GPU {gpu}] DONE compare {label} for target(s): {targets} -> {job['measure_json']}"
            gpu_status[gpu] = "idle"
        elif event == "error":
            failures += 1
            msg = f"🔴 [GPU {gpu}] FAIL compare {label} for target(s): {targets} - {res['error']}"
            gpu_status[gpu] = "idle"
        else:
            continue

        if use_ansi:
            _print_above_progress(msg, gpu_status, num_lines)
        else:
            print(msg)

    # Clear the progress area
    if use_ansi:
        sys.stdout.write(f"\033[{num_lines}A")
        for _ in range(num_lines):
            sys.stdout.write("\033[2K\n")
        sys.stdout.write(f"\033[{num_lines}A")
        sys.stdout.flush()

    for p in procs:
        p.join()
    if failures:
        raise RuntimeError(f"Optimized comparative measure stage failed for {failures} job(s)")


def _run_optimized_opt_stage(
    model_dir: str,
    optimized_bpws: List[str],
    devices: List[int],
    layers: int = 2,
    write_logs: bool = True,
) -> None:
    if not optimized_bpws:
        return

    measure_script, optimize_script = _resolve_exllamav3_util_scripts()
    compare_jobs, optimize_jobs = _build_optimized_jobs(model_dir, optimized_bpws)

    queued_jobs: List[dict] = []
    for job in compare_jobs:
        label = f"{job['low']}-{job['high']}"
        if os.path.exists(job["measure_json"]):
            print(
                f"🟦 skipping comparative measure {label}: {os.path.basename(job['measure_json'])} already exists"
            )
            continue
        queued_jobs.append(job)

    _run_optimized_compare_queue(
        model_dir=model_dir,
        compare_jobs=queued_jobs,
        devices=devices,
        measure_script=measure_script,
        layers=layers,
        write_logs=write_logs,
    )

    for job in optimize_jobs:
        frac = job["optimized"]
        out_dir = job["out_dir"]
        if os.path.isdir(out_dir) and os.path.isfile(os.path.join(out_dir, "config.json")):
            print(f"🟦 skipping optimized optimize {frac}: output already exists")
            continue
        optimize_cmd = [
            sys.executable,
            optimize_script,
            "-m",
            job["measure_json"],
            "-o",
            out_dir,
            "-b",
            frac,
        ]
        print(f"\n⚙️ Optimizing optimized quant {frac}")
        _run_cmd(optimize_cmd)


def _bpw_sort_key(w: str):
    if w == "bf16":
        return (-1.0, w)
    try:
        return (float(w), w)
    except Exception:
        return (1e9, w)


def _task_to_csv_label(task: str) -> str:
    """Map an internal measurement task label to its CSV weights label."""
    return "bf16" if task == "base" else task


def _filter_measure_tasks_for_checkpoint(requested_tasks: List[str], existing_labels: set[str]) -> List[str]:
    """Filter measurement tasks based on existing canonical CSV labels."""
    return [task for task in requested_tasks if _task_to_csv_label(task) not in existing_labels]


def _merge_csvs(out_csv: str, shard_csvs: List[str]) -> None:
    """Merge existing output CSV plus shard CSVs into *out_csv*.

    Uses field-level merge: for each label, non-empty field values from later
    sources overwrite earlier ones.  This lets a KL-only row and a PPL-only row
    combine into a single complete row.
    """
    fieldnames = ["weights", "KL Div", "PPL r-100", "GiB"]
    rows = {}
    sources = [out_csv, *shard_csvs]

    for path in sources:
        if not os.path.exists(path):
            continue
        with open(path, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                w = (row.get("weights") or "").strip()
                if not w:
                    continue
                if w not in rows:
                    rows[w] = dict(row)
                else:
                    for field in fieldnames:
                        new_val = (row.get(field) or "").strip()
                        if new_val:
                            rows[w][field] = new_val

    # Write merged
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key in sorted(rows.keys(), key=_bpw_sort_key):
            w.writerow(rows[key])


def _read_csv_rows(csv_path: str) -> Dict[str, dict]:
    """Read CSV into ``{label: row_dict}``."""
    out: Dict[str, dict] = {}
    if not os.path.exists(csv_path):
        return out
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = (row.get("weights") or "").strip()
            if w:
                out[w] = dict(row)
    return out


# ---------------------------------------------------------------------------
# Synthetic progress for measure subprocesses (ppl_layer / model_diff)
# ---------------------------------------------------------------------------

_TOTAL_LAYERS_RE = re.compile(r"Processing\s+(\d+)\s+layers", re.IGNORECASE)
_LAYER_LINE_RE = re.compile(r"^\s*--\s+.*\s{2,}(?:time:|rfn_err:)")
_RESULT_LINE_RE = re.compile(r"Perplexity:|KL divergence", re.IGNORECASE)


def _run_measure_subprocess(
    cmd: List[str],
    device: int,
    results: "Queue[Optional[dict]]",
    phase_label: str,
    log_f: Optional[IO] = None,
) -> str:
    """Run a measure subprocess, parse layer output, and send synthetic
    progress bar events through *results*.

    Returns the full captured output as a string.
    """
    if log_f:
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    assert proc.stdout is not None

    buf_lines: List[str] = []
    total_layers: Optional[int] = None
    completed = 0
    last_send: float = 0.0

    for line in proc.stdout:
        buf_lines.append(line)
        if log_f:
            log_f.write(line)
            log_f.flush()

        # Detect total layer count
        if total_layers is None:
            m = _TOTAL_LAYERS_RE.search(line)
            if m:
                total_layers = int(m.group(1))

        # Detect layer completion (lines like " -- model.layers.0.attn  ...")
        if total_layers and _LAYER_LINE_RE.match(line):
            completed += 1
            if completed == 1:
                # First layer (embed) → 10%
                pct = 10
            elif completed < total_layers:
                # Regular layers → 10-90%
                mid_total = max(total_layers - 2, 1)
                mid_done = completed - 1
                pct = 10 + int((mid_done / mid_total) * 80)
            else:
                # Head/logits layer finished
                pct = 100

            now = time.monotonic()
            if now - last_send >= 0.5 or pct >= 100:
                bar = _build_synthetic_bar(pct)
                results.put({
                    "event": "progress",
                    "device": device,
                    "text": f"{phase_label} {bar} ({completed}/{total_layers})",
                })
                last_send = now

        # Detect final result lines → jump to 100%
        if total_layers and _RESULT_LINE_RE.search(line) and completed < total_layers:
            bar = _build_synthetic_bar(100)
            results.put({
                "event": "progress",
                "device": device,
                "text": f"{phase_label} {bar} ({total_layers}/{total_layers})",
            })
            last_send = time.monotonic()

    proc.wait()
    if proc.returncode != 0:
        full_out = "".join(buf_lines)
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}\n\n"
            f"Output:\n{full_out}"
        )
    return "".join(buf_lines)


_KL_RE = re.compile(
    r"(?:KL|K/L)\s+divergence(?:\s+\(A,\s+B\))?:\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)",
    re.IGNORECASE,
)
_PPL_RE = re.compile(
    r"Perplexity:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Catbench progress parsing
# ---------------------------------------------------------------------------

_CATBENCH_LOADED_RE = re.compile(r"CATBENCH_MODEL_LOADED")
_CATBENCH_SAMPLE_RE = re.compile(r"CATBENCH_SAMPLE_DONE\s+(\d+)/(\d+)")
_CATBENCH_SAMPLE_START_RE = re.compile(r"CATBENCH_SAMPLE_START\s+(\d+)/(\d+)")
_CATBENCH_TOKENS_RE = re.compile(r"CATBENCH_TOKENS\s+(\d+)\s+([\d.]+)")


def _run_catbench_subprocess(
    cmd: List[str],
    device: int,
    results: "Queue[Optional[dict]]",
    phase_label: str,
    log_f: Optional[IO] = None,
    cuda_visible_devices: Optional[str] = None,
) -> str:
    """Run a catbench subprocess and send progress events.

    Progress model:
      - 0-100% bar during model loading
      - Token count display during inference
      - Sample completion markers
    """
    if log_f:
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices or str(device)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    assert proc.stdout is not None

    buf_lines: List[str] = []
    last_send: float = 0.0
    current_sample = ""
    model_loaded = False
    load_lines = 0

    # Show initial loading progress bar
    bar = _build_synthetic_bar(0)
    results.put({
        "event": "progress",
        "device": device,
        "text": f"{phase_label} {bar} (loading)",
    })

    for line in proc.stdout:
        buf_lines.append(line)
        if log_f:
            log_f.write(line)
            log_f.flush()

        # Model loaded → 100% bar
        if _CATBENCH_LOADED_RE.search(line):
            model_loaded = True
            bar = _build_synthetic_bar(100)
            results.put({
                "event": "progress",
                "device": device,
                "text": f"{phase_label} {bar} (loaded)",
            })
            last_send = time.monotonic()
            continue

        # During loading phase, increment bar for each output line
        if not model_loaded:
            load_lines += 1
            pct = min(95, load_lines * 3)
            now = time.monotonic()
            if now - last_send >= 0.3:
                bar = _build_synthetic_bar(pct)
                results.put({
                    "event": "progress",
                    "device": device,
                    "text": f"{phase_label} {bar} (loading)",
                })
                last_send = now
            continue

        # Sample start → update current sample label
        m = _CATBENCH_SAMPLE_START_RE.search(line)
        if m:
            current_sample = f"{m.group(1)}/{m.group(2)}"
            results.put({
                "event": "progress",
                "device": device,
                "text": f"{phase_label} | sample {current_sample} | 0 tokens",
            })
            last_send = time.monotonic()
            continue

        # Token progress → show count and TPS
        m = _CATBENCH_TOKENS_RE.search(line)
        if m:
            tokens = m.group(1)
            tps = m.group(2)
            now = time.monotonic()
            if now - last_send >= 0.3:
                results.put({
                    "event": "progress",
                    "device": device,
                    "text": f"{phase_label} | sample {current_sample} | {tokens} tokens ({tps} t/s)",
                })
                last_send = now
            continue

        # Sample done
        m = _CATBENCH_SAMPLE_RE.search(line)
        if m:
            i_done = int(m.group(1))
            n_total = int(m.group(2))
            results.put({
                "event": "progress",
                "device": device,
                "text": f"{phase_label} | sample {i_done}/{n_total} done",
            })
            last_send = time.monotonic()

    proc.wait()
    if proc.returncode != 0:
        full_out = "".join(buf_lines)
        raise RuntimeError(
            f"Catbench failed with exit code {proc.returncode}: {' '.join(cmd)}\n\n"
            f"Output:\n{full_out}"
        )
    return "".join(buf_lines)


def _worker_measure(
    base_dir: str,
    device: int,
    csv_path: str,
    tasks: "Queue[Optional[dict]]",
    results: "Queue[Optional[dict]]",
    log_path: Optional[str],
    ppl_rows: int = 100,
) -> None:
    """Phase-agnostic worker.  Each task is a dict with keys ``label`` and
    ``phase`` (``"kl"`` or ``"ppl"``).  ``None`` is the termination sentinel.
    """
    log_f = None
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_f = open(log_path, "w")

    ensure_csv_exists(csv_path)

    model_diff_script = find_model_diff_script()

    while True:
        job = tasks.get()
        if job is None:
            results.put(None)  # Sentinel
            break

        task_label = job["label"]
        phase = job["phase"]
        label = "bf16" if task_label == "base" else str(task_label)
        model_dir = base_dir if task_label == "base" else os.path.join(base_dir, str(task_label))
        phase_tag = phase.upper()
        results.put({"event": "start", "device": device, "label": label, "phase": phase})

        try:
            if phase == "kl":
                # --- KL divergence ---
                kl_cmd = [
                    sys.executable,
                    model_diff_script,
                    "-ma", base_dir,
                    "-mb", model_dir,
                    "-r", "10",
                    "-d", str(device),
                ]
                kl_out = _run_measure_subprocess(kl_cmd, device, results, f"{label} KL", log_f)
                kl_match = _KL_RE.search(kl_out)
                if not kl_match:
                    raise ValueError(
                        "Could not parse model_diff output (KL Divergence pattern did not match)."
                    )
                kl_div = float(kl_match.group(1))

                row = {
                    "weights": label,
                    "KL Div": kl_div,
                    "PPL r-100": "",
                    "GiB": file_size_gib(model_dir),
                }
                append_csv_row(csv_path, row)
                results.put({
                    "event": "done", "device": device, "label": label,
                    "phase": phase, "row": row,
                })

            elif phase == "ppl":
                # --- PPL ---
                ppl_cmd = [
                    sys.executable,
                    "-m", "ezexl3.ppl_layer",
                    "-m", model_dir,
                    "-r", str(ppl_rows),
                    "-d", str(device),
                ]
                ppl_out = _run_measure_subprocess(ppl_cmd, device, results, f"{label} PPL", log_f)
                ppl_match = _PPL_RE.search(ppl_out)
                if not ppl_match:
                    raise ValueError(
                        "Could not parse ppl_layer output (Perplexity pattern didn't match)."
                    )
                ppl_val = float(ppl_match.group(1))

                # For base, hardcode KL=0.0; for others leave blank (merge fills it)
                kl_field = 0.0 if task_label == "base" else ""
                row = {
                    "weights": label,
                    "KL Div": kl_field,
                    "PPL r-100": ppl_val,
                    "GiB": file_size_gib(model_dir),
                }
                append_csv_row(csv_path, row)
                results.put({
                    "event": "done", "device": device, "label": label,
                    "phase": phase, "row": row,
                })

            elif phase == "catbench":
                # --- Catbench ---
                n_samples = job.get("n_samples", 3)
                catbench_out_dir = os.path.join(base_dir, "catbench")
                catbench_cmd = [
                    sys.executable,
                    "-m", "ezexl3.catbench",
                    "-m", model_dir,
                    "-cs", str(4096 + 512),
                    "-n", str(n_samples),
                    "-o", catbench_out_dir,
                    "-l", label,
                ]
                _run_catbench_subprocess(catbench_cmd, device, results, f"{label} CAT", log_f)
                results.put({
                    "event": "done", "device": device, "label": label,
                    "phase": phase, "row": {},
                })

        except Exception as e:
            import traceback
            if log_f:
                traceback.print_exc(file=log_f)
                log_f.flush()
            results.put({
                "event": "error", "device": device, "label": label,
                "phase": phase, "error": str(e),
            })

    if log_f:
        log_f.close()




def _parse_measure_args(measure_args: List[str], default_devices: List[int]) -> tuple[int, List[int]]:
    """Parse passthrough measure args supported by ezexl3.

    Supported:
      -r/--rows <int>         # PPL rows
      -d/--device/--devices   # CUDA device list (comma-separated)
    """
    ppl_rows = 100
    devices = list(default_devices)

    i = 0
    while i < len(measure_args):
        tok = measure_args[i]

        if tok in ("-r", "--rows"):
            if i + 1 >= len(measure_args):
                raise ValueError("Missing value for --measure-args -r/--rows")
            try:
                ppl_rows = int(measure_args[i + 1])
                if ppl_rows <= 0:
                    raise ValueError("--measure-args rows must be > 0")
            except ValueError as e:
                raise ValueError(f"Invalid rows value for --measure-args: {measure_args[i + 1]}") from e
            i += 2
            continue

        if tok in ("-d", "--device", "--devices"):
            if i + 1 >= len(measure_args):
                raise ValueError("Missing value for --measure-args -d/--device")
            val = measure_args[i + 1]
            parsed = [x.strip() for x in str(val).split(",") if x.strip()]
            if not parsed:
                raise ValueError("Empty device list in --measure-args -d/--device")
            try:
                devices = [int(x) for x in parsed]
            except ValueError as e:
                raise ValueError(f"Invalid device list for --measure-args: {val}") from e
            i += 2
            continue

        raise ValueError(
            f"Unsupported --measure-args token: {tok}. Supported flags: -r/--rows, -d/--device/--devices"
        )

    return ppl_rows, devices


def run_quant_stage(
    model_dir: str,
    bpws: List[str],
    devices: List[int],
    device_ratios: Optional[str],
    quant_args: List[str],
    out_template: str = "{model}/{bpw}",
    w_template: str = "{model}/w-{bpw}",
    dry_run: bool = False,
    continue_on_error: bool = False,
    optimized_measure_layers: int = 2,
) -> int:
    if optimized_measure_layers not in (1, 2, 3):
        raise ValueError("optimized_measure_layers must be one of: 1, 2, 3")

    model_dir = os.path.abspath(model_dir)
    bpws = [str(b) for b in bpws]
    devices = list(devices)


    models = [model_dir]
    forwarded = list(quant_args)

    # Inject devices if not already in quant_args
    if devices and ("-d" not in forwarded and "--devices" not in forwarded):
        devices_str = ",".join(str(d) for d in devices)
        forwarded += ["-d", devices_str]

    # If device ratios were supplied via main flags, and user didn't also pass -dr in quant_args,
    # we can inject it (but keep it minimal and non-magical).
    if device_ratios and ("-dr" not in forwarded and "--device-ratios" not in forwarded):
        forwarded += ["-dr", device_ratios]

    rc = quant_run(
        models=models,
        bpws=bpws,
        forwarded=forwarded,
        out_template=out_template,
        w_template=w_template,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
    )
    return rc


def run_measure_stage(
    model_dir: str,
    bpws: List[str],
    devices: List[int],
    write_logs: bool = True,
    measure_args: Optional[List[str]] = None,
    catbench_n: int = 0,
) -> int:
    model_dir = os.path.abspath(model_dir)
    bpws = [str(b) for b in bpws]
    devices = list(devices)
    ppl_rows, devices = _parse_measure_args(measure_args or [], devices)
    if not devices:
        raise ValueError("No CUDA devices available for measure stage. Provide -d/--devices.")

    out_csv = default_csv_path(model_dir)

    # Shard CSVs
    shard_csvs = []
    log_paths = []
    for d in devices:
        shard_csvs.append(os.path.join(model_dir, f"{os.path.basename(model_dir)}Measured.gpu{d}.csv"))
        log_paths.append(os.path.join(model_dir, "logs", f"measure_gpu{d}.log") if write_logs else None)

    # Seed merged output from any existing shard state.
    _merge_csvs(out_csv, shard_csvs)

    # Per-field checkpointing: read merged CSV and decide which phases to skip.
    existing_rows = _read_csv_rows(out_csv)

    kl_tasks: List[dict] = []
    ppl_tasks: List[dict] = []

    for bpw in bpws:
        label = _task_to_csv_label(bpw)
        row = existing_rows.get(label, {})
        has_kl = bool((row.get("KL Div") or "").strip())
        has_ppl = bool((row.get("PPL r-100") or "").strip())

        # base never needs KL (hardcoded to 0.0)
        if bpw != "base" and not has_kl:
            kl_tasks.append({"label": bpw, "phase": "kl"})
        if not has_ppl:
            ppl_tasks.append({"label": bpw, "phase": "ppl"})

    # Always include base PPL if not yet measured
    base_label = "bf16"
    base_row = existing_rows.get(base_label, {})
    if not bool((base_row.get("PPL r-100") or "").strip()):
        if not any(t["label"] == "base" for t in ppl_tasks):
            ppl_tasks.append({"label": "base", "phase": "ppl"})

    # --- Catbench tasks ---
    catbench_tasks: List[dict] = []
    multi_gpu_catbench_tasks: List[dict] = []

    if catbench_n > 0:
        catbench_out_dir = os.path.join(model_dir, "catbench")

        # Build catbench tasks for all bpws
        for bpw in bpws:
            label = _task_to_csv_label(bpw)
            file_prefix = _catbench_file_prefix(label)
            if not _catbench_has_output(catbench_out_dir, file_prefix):
                catbench_tasks.append({
                    "label": bpw, "phase": "catbench", "n_samples": catbench_n,
                })

        # Include bf16 baseline
        if not _catbench_has_output(catbench_out_dir, "bf16"):
            catbench_tasks.append({
                "label": "base", "phase": "catbench", "n_samples": catbench_n,
            })

        # VRAM pre-flight: sort into multi-GPU vs single-GPU
        if len(devices) > 1 and catbench_tasks:
            from ezexl3.catbench import check_vram_fit, check_multi_gpu_fit
            single_gpu = []
            for task in catbench_tasks:
                task_label = task["label"]
                task_model_dir = model_dir if task_label == "base" else os.path.join(model_dir, str(task_label))
                fits, model_gib, avail_gib = check_vram_fit(task_model_dir, devices[0])
                if fits:
                    single_gpu.append(task)
                else:
                    # Check if it fits across all GPUs combined
                    multi_fits, model_gib, total_avail = check_multi_gpu_fit(task_model_dir, devices)
                    if multi_fits:
                        device_str = ",".join(str(d) for d in devices)
                        task["device_str"] = device_str
                        multi_gpu_catbench_tasks.append(task)
                    else:
                        task_disp = "bf16" if task_label == "base" else str(task_label)
                        print(f"  ⚠️  Skipping catbench for {task_disp}: "
                              f"{model_gib:.1f} GiB model won't fit "
                              f"({total_avail:.1f} GiB available across {len(devices)} GPUs)")
            catbench_tasks = single_gpu

    total_jobs = len(kl_tasks) + len(ppl_tasks) + len(catbench_tasks) + len(multi_gpu_catbench_tasks)

    if total_jobs == 0:
        print("✅ All requested measurement phases already exist. Nothing to do.")
        return 0

    n_kl = len(kl_tasks)
    n_ppl = len(ppl_tasks)
    n_cat = len(catbench_tasks) + len(multi_gpu_catbench_tasks)

    if n_kl < len(bpws) or n_ppl < len(bpws) + 1:
        print(f"ℹ️ Measurement checkpoint: {n_kl} KL + {n_ppl} PPL jobs remaining.")

    # --- Run multi-GPU catbench jobs first (sequentially, all GPUs) ---
    if multi_gpu_catbench_tasks:
        import threading
        from queue import Queue as TQueue

        mgpu_use_ansi = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

        print(f"\n🐱 Running {len(multi_gpu_catbench_tasks)} multi-GPU catbench job(s)...")

        mgpu_status: Dict[int, str] = {d: "idle" for d in devices}
        mgpu_num_lines = len(devices)

        # Print initial progress area
        if mgpu_use_ansi:
            for d in sorted(mgpu_status):
                sys.stdout.write(f"\033[2K  GPU {d} | idle\n")
            sys.stdout.flush()

        for task in multi_gpu_catbench_tasks:
            task_label = task["label"]
            label = "bf16" if task_label == "base" else str(task_label)
            task_model_dir = model_dir if task_label == "base" else os.path.join(model_dir, str(task_label))
            device_str = task["device_str"]
            catbench_cmd = [
                sys.executable, "-m", "ezexl3.catbench",
                "-m", task_model_dir,
                "-gs", ",".join("99" for _ in device_str.split(",")),
                "-cs", str(4096 + 512),
                "-n", str(task.get("n_samples", 3)),
                "-o", os.path.join(model_dir, "catbench"),
                "-l", label,
            ]
            phase_label = f"{label} CAT"
            mgpu_results: TQueue = TQueue()
            mgpu_error: List[Optional[Exception]] = [None]

            def _run_mgpu_catbench(
                _cmd: List[str] = catbench_cmd,
                _dev: int = devices[0],
                _q: TQueue = mgpu_results,
                _pl: str = phase_label,
                _cvd: str = device_str,
            ) -> None:
                try:
                    _run_catbench_subprocess(_cmd, _dev, _q, _pl,
                                            cuda_visible_devices=_cvd)
                except Exception as exc:
                    mgpu_error[0] = exc
                _q.put(None)

            t = threading.Thread(target=_run_mgpu_catbench)
            t.start()

            # Consume progress events and render to GPU status lines
            while True:
                ev = mgpu_results.get()
                if ev is None:
                    break
                if ev["event"] == "progress":
                    for d in devices:
                        mgpu_status[d] = ev["text"]
                    if mgpu_use_ansi:
                        _clear_and_redraw_progress(mgpu_status, mgpu_num_lines)

            t.join()

            if mgpu_error[0] is not None:
                msg = f"🔴 Multi-GPU catbench failed for {label}: {mgpu_error[0]}"
            else:
                msg = f"🐱 DONE {label} CATBENCH (multi-GPU [{device_str}])"

            for d in devices:
                mgpu_status[d] = "idle"
            if mgpu_use_ansi:
                _print_above_progress(msg, mgpu_status, mgpu_num_lines)
            else:
                print(msg)

        # Clear progress area
        if mgpu_use_ansi:
            sys.stdout.write(f"\033[{mgpu_num_lines}A")
            for _ in range(mgpu_num_lines):
                sys.stdout.write("\033[2K\n")
            sys.stdout.write(f"\033[{mgpu_num_lines}A")
            sys.stdout.flush()

    # Remaining jobs (KL + PPL + single-GPU catbench) into shared queue
    remaining_jobs = len(kl_tasks) + len(ppl_tasks) + len(catbench_tasks)
    if remaining_jobs == 0:
        if multi_gpu_catbench_tasks:
            print("✅ All catbench jobs complete (multi-GPU only).")
        return 0

    # Queue: all KL first, then all PPL, then catbench, then sentinels
    tasks: Queue = Queue()
    results: Queue = Queue()

    for t in kl_tasks:
        tasks.put(t)
    for t in ppl_tasks:
        tasks.put(t)
    for t in catbench_tasks:
        tasks.put(t)
    for _ in devices:
        tasks.put(None)

    procs: List[Process] = []
    for d, csvp, logp in zip(devices, shard_csvs, log_paths):
        p = Process(target=_worker_measure, args=(model_dir, d, csvp, tasks, results, logp, ppl_rows))
        p.daemon = False
        p.start()
        procs.append(p)
        time.sleep(2.0)

    cat_msg = f" + {n_cat} CAT" if n_cat else ""
    print(f"\n🚀 Measuring {n_kl} KL + {n_ppl} PPL{cat_msg} jobs on {len(devices)} GPUs...")

    use_ansi = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    gpu_status: Dict[int, str] = {d: "idle" for d in devices}
    num_lines = len(devices)

    # Print initial progress area (one line per GPU)
    if use_ansi:
        for d in sorted(gpu_status):
            sys.stdout.write(f"\033[2K  GPU {d} | idle\n")
        sys.stdout.flush()

    # Result listener loop
    active_workers = len(devices)
    failures = 0

    while active_workers > 0:
        res = results.get()
        if res is None:
            active_workers -= 1
            continue

        gpu = res["device"]
        event = res["event"]

        if event == "progress":
            gpu_status[gpu] = res["text"]
            if use_ansi:
                _clear_and_redraw_progress(gpu_status, num_lines)
            continue

        label = res["label"]
        phase = res.get("phase", "")
        phase_tag = phase.upper()

        if event == "start":
            msg = f"🧪 [GPU {gpu}] START {label} {phase_tag}"
            gpu_status[gpu] = f"{label} {phase_tag} | starting..."
        elif event == "done":
            row = res["row"]
            if phase == "kl":
                kl_val = row.get("KL Div", "N/A")
                msg = f"✅ [GPU {gpu}] DONE {label} KL: KL={kl_val}"
            elif phase == "catbench":
                msg = f"🐱 [GPU {gpu}] DONE {label} CATBENCH"
            else:
                ppl_val = row.get("PPL r-100", "N/A")
                msg = f"✅ [GPU {gpu}] DONE {label} PPL: PPL={ppl_val}"
            gpu_status[gpu] = "idle"
            if phase != "catbench":
                _merge_csvs(out_csv, shard_csvs)
        elif event == "error":
            failures += 1
            msg = f"🔴 [GPU {gpu}] FAIL {label} {phase_tag}: {res['error']}"
            gpu_status[gpu] = "idle"
        else:
            continue

        if use_ansi:
            _print_above_progress(msg, gpu_status, num_lines)
        else:
            print(msg)

    # Clear the progress area
    if use_ansi:
        sys.stdout.write(f"\033[{num_lines}A")
        for _ in range(num_lines):
            sys.stdout.write("\033[2K\n")
        sys.stdout.write(f"\033[{num_lines}A")
        sys.stdout.flush()

    for p in procs:
        p.join()

    # Final merge after all workers exit
    _merge_csvs(out_csv, shard_csvs)

    if failures:
        print(f"⚠️ Measurement stage completed with {failures} failure(s). Merged CSV: {out_csv}")
    else:
        print(f"✅ All measurements complete. Merged CSV: {out_csv}")
    return 0


def run_repo(
    model_dir: str,
    bpws: List[str],
    devices: List[int],
    device_ratios: Optional[str],
    quant_args: List[str],
    measure_args: List[str],  # reserved for later; keep but unused in v0
    do_quant: bool = True,
    do_measure: bool = True,
    do_readme: bool = True,
    cleanup: bool = False,
    write_logs: bool = True,
    interactive: bool = True,
    template: Optional[str] = None,
    include_graph: bool = True,
    include_measurements: bool = True,
    optimized_measure_layers: int = 2,
    catbench_n: int = 0,
) -> int:
    bpw_plan = _plan_repo_bpws(bpws)
    quant_bpws = bpw_plan["quant_integer_queue"]
    optimized_bpws = bpw_plan["requested_optimizeds"]
    measure_bpws = bpw_plan["measure_queue"]

    auto_added = [b for b in quant_bpws if b not in bpw_plan["requested_integers"]]
    if auto_added:
        print(
            "ℹ️ Added required integer quants for optimized targets: "
            + ", ".join(auto_added)
        )

    # --- Stage 1: quantize ---
    if do_quant:
        rc = run_quant_stage(
            model_dir=model_dir,
            bpws=quant_bpws,
            devices=devices,
            device_ratios=device_ratios,
            quant_args=quant_args,
        )
        if rc != 0:
            return rc

    # --- Stage 2: optimized optimize ---
    if do_quant and optimized_bpws:
        _run_optimized_opt_stage(
            model_dir=model_dir,
            optimized_bpws=optimized_bpws,
            devices=devices,
            layers=optimized_measure_layers,
            write_logs=write_logs,
        )

    # --- Stage 3: measure (sharded, dynamic queue) ---
    if do_measure or catbench_n > 0:
        rc = run_measure_stage(
            model_dir=model_dir,
            bpws=measure_bpws,
            devices=devices,
            write_logs=write_logs,
            measure_args=measure_args,
            catbench_n=catbench_n,
        )
        if rc != 0:
            return rc

    # --- Stage 4: README generation ---
    if do_readme:
        from ezexl3.readme import run_readme
        print("Generating README...")
        run_readme(
            model_dir,
            template_name=template,
            interactive=interactive,
            include_graph=include_graph,
            include_measurements=include_measurements,
            bpws_hint=measure_bpws,
            include_catbench=(catbench_n > 0),
        )

    # --- Stage 5: cleanup ---
    if cleanup:
        import shutil
        import glob
        print("\n🧹 Cleaning up working directories and temporary files...")
        
        # 1. w-* dirs
        w_dirs = glob.glob(os.path.join(model_dir, "w-*"))
        for d in w_dirs:
            if os.path.isdir(d):
                print(f"  Removing workspace {os.path.basename(d)}...")
                try: shutil.rmtree(d)
                except Exception as e: print(f"  🔴 Failed to remove {d}: {e}")
        
        # 2. *.gpu*.csv
        gpu_csvs = glob.glob(os.path.join(model_dir, "*.gpu*.csv"))
        for f in gpu_csvs:
            print(f"  Removing shard CSV {os.path.basename(f)}...")
            try: os.remove(f)
            except Exception as e: print(f"  🔴 Failed to remove {f}: {e}")
            
        # 3. logs/
        logs_dir = os.path.join(model_dir, "logs")
        if os.path.isdir(logs_dir):
            print(f"  Removing logs directory...")
            try: shutil.rmtree(logs_dir)
            except Exception as e: print(f"  🔴 Failed to remove {logs_dir}: {e}")
            
        print("✅ Cleanup complete.")

    return 0
