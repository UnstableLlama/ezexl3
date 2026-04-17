from __future__ import annotations

import multiprocessing as _mp
import os
import pty
import re
import select
import subprocess
import sys
import threading
import time
from typing import IO, List, Optional

from ezexl3.repo_progress import _build_synthetic_bar, _strip_ansi


# ---------------------------------------------------------------------------
# Isolated quantize helper (interleaved pipeline)
# ---------------------------------------------------------------------------
# Quantization calls convert_main() in-process, which initialises a CUDA
# context and leaves exllamav3 tensors/allocator caches pinned in the parent
# for the rest of its life. In the interleaved pipeline the very next stage is
# a KL verification subprocess that loads two models side-by-side on the same
# GPU — with ~5 GiB of leftover parent state it OOMs. Running quantize in a
# spawned subprocess lets the OS reclaim 100% of that memory on exit.
def _quant_worker_entry(model_dir, bpw, forwarded, out_tmpl, w_tmpl, result_q):
    try:
        # Import inside the child so the parent never has to fork-after-CUDA.
        from ezexl3.quantize import run_one
        ok = run_one(
            model_dir, bpw, forwarded,
            out_tmpl=out_tmpl, w_tmpl=w_tmpl, dry_run=False,
        )
        result_q.put(("ok", bool(ok)))
    except BaseException as e:  # noqa: BLE001
        import traceback
        result_q.put(("error", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def _run_quant_one_isolated(
    model_dir: str,
    bpw: str,
    forwarded: List[str],
    out_tmpl: str,
    w_tmpl: str,
) -> bool:
    """Run quant_run_one in a spawned subprocess so GPU memory is fully
    released before the next verification stage runs.

    Returns True on success, False on failure.
    """
    ctx = _mp.get_context("spawn")
    result_q = ctx.Queue()
    p = ctx.Process(
        target=_quant_worker_entry,
        args=(model_dir, bpw, forwarded, out_tmpl, w_tmpl, result_q),
    )
    p.start()
    p.join()

    # Drain any result the child may have placed on the queue before inspecting
    # the exit code so we can surface crash tracebacks even on hard failures.
    payload = None
    status = None
    try:
        if not result_q.empty():
            status, payload = result_q.get_nowait()
    except Exception:
        pass

    if p.exitcode != 0:
        if status == "error" and payload:
            print(f"🔴 Quantize worker crashed (exit={p.exitcode}):\n{payload}")
        else:
            print(f"🔴 Quantize worker crashed with exit code {p.exitcode}")
        return False

    if status is None:
        print("🔴 Quantize worker exited cleanly but sent no result")
        return False
    if status == "error":
        print(f"🔴 Quantize worker error:\n{payload}")
        return False
    return bool(payload)


def _run_cmd(cmd: List[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")


def _run_cmd_with_progress(
    cmd: List[str],
    device: int,
    results,
    log_f: Optional[IO] = None,
) -> str:
    """Run *cmd* in a PTY, stream output to *log_f*, and send throttled progress events."""
    if log_f:
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.flush()

    master_fd: Optional[int] = None
    try:
        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(cmd, stdout=slave_fd, stderr=slave_fd, close_fds=True)
        os.close(slave_fd)
    except Exception:
        if master_fd is not None:
            os.close(master_fd)
            master_fd = None
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    last_send: float = 0.0
    buf = ""

    def _drain_fd(fd: int) -> bool:
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
        lines = buf.split("\n")
        tail = lines[-1]
        if "\r" in tail:
            candidate = tail.split("\r")[-1].strip()
        else:
            candidate = tail.strip()
        if candidate:
            results.put({"event": "progress", "device": device, "text": _strip_ansi(candidate)})
            last_send = now

    if master_fd is not None:
        while True:
            ready, _, _ = select.select([master_fd], [], [], 0.5)
            if ready:
                if not _drain_fd(master_fd):
                    break
            elif proc.poll() is not None:
                while _drain_fd(master_fd):
                    pass
                break
        os.close(master_fd)
    else:
        assert proc.stdout is not None
        while _drain_pipe():
            pass

    if proc.stdout is not None:
        proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return buf


_TOTAL_LAYERS_RE = re.compile(r"Processing\s+(\d+)\s+layers", re.IGNORECASE)
_LAYER_LINE_RE = re.compile(r"^\s*--\s+.*\s{2,}(?:time:|rfn_err:)")
_RESULT_LINE_RE = re.compile(r"Perplexity:|KL divergence", re.IGNORECASE)


def _run_measure_subprocess(
    cmd: List[str],
    device: int,
    results,
    phase_label: str,
    log_f: Optional[IO] = None,
) -> str:
    """Run a measure subprocess, parse layer output, and send synthetic progress events."""
    if log_f:
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None

    buf_lines: List[str] = []
    total_layers: Optional[int] = None
    completed = 0
    last_send: float = 0.0

    while True:
        line = proc.stdout.readline()
        if not line:
            break
        buf_lines.append(line)
        if log_f:
            log_f.write(line)
            log_f.flush()

        if total_layers is None:
            m = _TOTAL_LAYERS_RE.search(line)
            if m:
                total_layers = int(m.group(1))

        if total_layers and _LAYER_LINE_RE.match(line):
            completed += 1
            if completed == 1:
                pct = 10
            elif completed < total_layers:
                mid_total = max(total_layers - 2, 1)
                mid_done = completed - 1
                pct = 10 + int((mid_done / mid_total) * 80)
            else:
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

        if total_layers and _RESULT_LINE_RE.search(line) and completed < total_layers:
            bar = _build_synthetic_bar(100)
            results.put({
                "event": "progress",
                "device": device,
                "text": f"{phase_label} {bar} ({total_layers}/{total_layers})",
            })
            last_send = time.monotonic()

    proc.stdout.close()
    proc.wait()
    if proc.returncode != 0:
        full_out = "".join(buf_lines)
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}\n\nOutput:\n{full_out}"
        )
    return "".join(buf_lines)


_CATBENCH_LOADED_RE = re.compile(r"CATBENCH_MODEL_LOADED")
_CATBENCH_SAMPLE_RE = re.compile(r"CATBENCH_SAMPLE_DONE\s+(\d+)/(\d+)")
_CATBENCH_SAMPLE_START_RE = re.compile(r"CATBENCH_SAMPLE_START\s+(\d+)/(\d+)")
_CATBENCH_TOKENS_RE = re.compile(r"CATBENCH_TOKENS\s+(\d+)\s+([\d.]+)")


def _run_catbench_subprocess(
    cmd: List[str],
    device: int,
    results,
    phase_label: str,
    log_f: Optional[IO] = None,
    cuda_visible_devices: Optional[str] = None,
) -> str:
    """Run a catbench subprocess and send progress events."""
    if log_f:
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.flush()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices or str(device)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None

    buf_lines: List[str] = []
    last_send: float = 0.0
    last_heartbeat: float = 0.0
    current_sample = ""
    model_loaded = False
    load_start: float = time.monotonic()
    load_done = threading.Event()

    # Persistent log-line feedback so the UI terminal shows something every
    # few seconds during the (minutes-long) token-generation phase, in
    # addition to the in-place progress bar.
    #
    # Two channels:
    #  * _emit_log: one-off scrollback events (model loaded, sample N
    #    starting, informational lines from catbench.py).
    #  * _emit_progress: recurring per-token heartbeats that should overwrite
    #    a single line in the UI terminal. The leading "\rgpu{N}: " prefix
    #    lets terminal.js's _updateProgressLine group them per-GPU so each
    #    device gets its own updating line instead of cascading into
    #    scrollback.
    # Catbench is purely visual (no data rows, no CSV), so we funnel every
    # terminal update through a single \r-prefixed line per GPU instead of
    # interleaving scrollback events. terminal.js's _updateProgressLine
    # groups lines by the "gpuN:" key, so each device gets exactly one
    # line that cycles through states: loading → sample N | tokens → done.
    progress_key = f"gpu{device}:"
    heartbeat_interval = 2.0

    def _emit_progress(text: str) -> None:
        sys.stdout.write(f"\r{progress_key} {phase_label} {text}\n")
        sys.stdout.flush()

    def _loading_progress_ticker():
        while not load_done.wait(timeout=0.5):
            elapsed = time.monotonic() - load_start
            pct = int(95 * (1 - 2 ** (-elapsed / 30)))
            pct = max(pct, 1)
            bar = _build_synthetic_bar(pct)
            _emit_progress(f"loading {pct}%")
            results.put({
                "event": "progress",
                "device": device,
                "text": f"{phase_label} {bar} (loading)",
            })

    ticker = threading.Thread(target=_loading_progress_ticker, daemon=True)
    ticker.start()

    for line in proc.stdout:
        buf_lines.append(line)
        if log_f:
            log_f.write(line)
            log_f.flush()

        if _CATBENCH_LOADED_RE.search(line):
            model_loaded = True
            load_done.set()
            bar = _build_synthetic_bar(100)
            results.put({
                "event": "progress",
                "device": device,
                "text": f"{phase_label} {bar} (loaded)",
            })
            last_send = time.monotonic()
            _emit_progress("loaded")
            continue

        if not model_loaded:
            continue

        m = _CATBENCH_SAMPLE_START_RE.search(line)
        if m:
            current_sample = f"{m.group(1)}/{m.group(2)}"
            results.put({
                "event": "progress",
                "device": device,
                "text": f"{phase_label} | sample {current_sample} | 0 tokens",
            })
            last_send = time.monotonic()
            last_heartbeat = last_send
            _emit_progress(f"sample {current_sample} | starting")
            continue

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
            # Per-GPU updating heartbeat in the UI terminal so the user
            # sees live token counts without flooding scrollback.
            if now - last_heartbeat >= heartbeat_interval:
                _emit_progress(f"sample {current_sample} | {tokens} tokens ({tps} t/s)")
                last_heartbeat = now
            continue

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
            _emit_progress(f"sample {i_done}/{n_total} done")
            continue
        # All other informational stdout from catbench.py (model size,
        # cache info, per-sample wrap-up, SVG extraction notices, etc.)
        # is intentionally dropped from the UI terminal — the per-GPU
        # progress line above already reflects live state, and these
        # lines would otherwise cascade in scrollback. They still reach
        # buf_lines and the log file so post-mortem debugging is intact.

    proc.stdout.close()
    proc.wait()
    load_done.set()
    if proc.returncode != 0:
        full_out = "".join(buf_lines)
        raise RuntimeError(
            f"Catbench failed with exit code {proc.returncode}: {' '.join(cmd)}\n\nOutput:\n{full_out}"
        )
    return "".join(buf_lines)
