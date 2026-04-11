from __future__ import annotations

import math
import os
import sys
from multiprocessing import Process, Queue
from typing import IO, Callable, Dict, List, Optional, Tuple

from ezexl3.repo_progress import (
    _cleanup_gpu_progress,
    _init_gpu_progress,
    _print_msg_with_progress,
    _redraw_gpu_progress,
)
from ezexl3.repo_subprocess import _run_cmd, _run_cmd_with_progress

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")
_MEASURE_SCRIPT = os.path.join(_VENDOR_DIR, "measure.py")
_OPTIMIZE_SCRIPT = os.path.join(_VENDOR_DIR, "optimize.py")


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
    model_dir: str,
    device: int,
    layers: int,
    tasks,
    results,
    log_path: Optional[str],
    run_cmd_with_progress_fn: Callable = _run_cmd_with_progress,
    measure_script: str = _MEASURE_SCRIPT,
    executable: str = sys.executable,
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
                executable,
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
            run_cmd_with_progress_fn(cmd, device, results, log_f)
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
    layers: int,
    write_logs: bool = True,
    process_cls=None,
    queue_cls=None,
    worker_fn: Callable = _worker_optimized_compare,
    init_gpu_progress_fn: Callable = _init_gpu_progress,
    redraw_gpu_progress_fn: Callable = _redraw_gpu_progress,
    print_msg_with_progress_fn: Callable = _print_msg_with_progress,
    cleanup_gpu_progress_fn: Callable = _cleanup_gpu_progress,
) -> None:
    if not compare_jobs:
        return
    if not devices:
        raise ValueError("No CUDA devices available for optimized comparative measure stage")

    if process_cls is None:
        process_cls = Process
    if queue_cls is None:
        queue_cls = Queue

    tasks = queue_cls()
    results = queue_cls()

    for job in compare_jobs:
        tasks.put(job)
    for _ in devices:
        tasks.put(None)

    procs: List[Process] = []
    for device in devices:
        log_path = os.path.join(model_dir, "logs", f"optimized_compare_gpu{device}.log") if write_logs else None
        p = process_cls(
            target=worker_fn,
            args=(model_dir, device, layers, tasks, results, log_path),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    print(f"\n🚀 Optimized comparative measure: {len(compare_jobs)} jobs on {len(devices)} GPUs...")

    use_ansi = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    gpu_status: Dict[int, str] = {d: "idle" for d in devices}
    num_lines = len(devices)
    init_gpu_progress_fn(use_ansi, gpu_status)

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
            redraw_gpu_progress_fn(use_ansi, gpu_status, num_lines)
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

        print_msg_with_progress_fn(msg, use_ansi, gpu_status, num_lines)

    cleanup_gpu_progress_fn(use_ansi, num_lines)

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
    build_jobs_fn: Callable = _build_optimized_jobs,
    run_compare_queue_fn: Callable = _run_optimized_compare_queue,
    run_cmd_fn: Callable = _run_cmd,
    optimize_script: str = _OPTIMIZE_SCRIPT,
    executable: str = sys.executable,
) -> None:
    if not optimized_bpws:
        return

    compare_jobs, optimize_jobs = build_jobs_fn(model_dir, optimized_bpws)

    queued_jobs: List[dict] = []
    for job in compare_jobs:
        label = f"{job['low']}-{job['high']}"
        if os.path.exists(job["measure_json"]):
            print(f"🟦 skipping comparative measure {label}: {os.path.basename(job['measure_json'])} already exists")
            continue
        queued_jobs.append(job)

    run_compare_queue_fn(
        model_dir=model_dir,
        compare_jobs=queued_jobs,
        devices=devices,
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
            executable,
            optimize_script,
            "-m",
            job["measure_json"],
            "-o",
            out_dir,
            "-b",
            frac,
        ]
        print(f"\n⚙️ Optimizing optimized quant {frac}")
        run_cmd_fn(optimize_cmd)
