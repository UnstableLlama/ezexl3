# ezexl3/repo.py
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import List, Optional, Tuple

from ezexl3.quantize import run as quant_run
from ezexl3.measure import run_measure, default_csv_path, read_existing_weights


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _bpw_sort_key(w: str):
    if w == "bf16":
        return (-1.0, w)
    try:
        return (float(w), w)
    except Exception:
        return (1e9, w)


def _merge_csvs(out_csv: str, shard_csvs: List[str]) -> None:
    # Merge by unique weights, keeping first occurrence (should be deterministic if bf16 is in shard0)
    rows = {}
    for p in shard_csvs:
        if not os.path.exists(p):
            continue
        with open(p, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                w = (row.get("weights") or "").strip()
                if not w:
                    continue
                if w not in rows:
                    rows[w] = row

    # Write merged
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = ["weights", "PPL r-10", "K/L Div", "PPL r-100", "GiB"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for key in sorted(rows.keys(), key=_bpw_sort_key):
            w.writerow(rows[key])


def _worker_measure(
    base_dir: str,
    device: int,
    csv_path: str,
    tasks: "Queue[str]",
    log_path: Optional[str],
    exllamav3_root: Optional[str] = None,
) -> None:
    # Optional per-worker log file
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_f = open(log_path, "w")
        sys.stdout = log_f  # type: ignore
        sys.stderr = log_f  # type: ignore

    # Make sure shard CSV exists early
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    while True:
        item = tasks.get()
        if item is None:
            break

        # Each item is a single quant label ("base" or "2" etc.)
        run_measure(
            base_dir=base_dir,
            quants=[item],
            device=device,
            csv_path=csv_path,
            exllamav3_root=exllamav3_root,
            skip_done=True,
        )

    if log_path:
        sys.stdout.flush()
        sys.stderr.flush()
        log_f.close()


def run_repo(
    model_dir: str,
    bpws: List[str],
    devices: List[int],
    device_ratios: Optional[str],
    quant_args: List[str],
    measure_args: List[str],  # reserved for later; keep but unused in v0
    exllamav3_root=None,
    do_quant: bool = True,
    do_measure: bool = True,
    do_report: bool = False,  # report stage stub for now
    cleanup: bool = False,
    write_logs: bool = True,
) -> int:
    model_dir = os.path.abspath(model_dir)
    bpws = [str(b) for b in bpws]
    devices = list(devices)

    # --- Stage 1: quantize ---
    if do_quant:
        models = [model_dir]
        forwarded = list(quant_args)

        # If device ratios were supplied via main flags, and user didn't also pass -dr in quant_args,
        # we can inject it (but keep it minimal and non-magical).
        if device_ratios and ("-dr" not in forwarded and "--device-ratios" not in forwarded):
            forwarded += ["-dr", device_ratios]

        rc = quant_run(
            models=models,
            bpws=bpws,
            forwarded=forwarded,
            dry_run=False,
            continue_on_error=False,
        )
        if rc != 0:
            return rc

    # --- Stage 2: measure (sharded, dynamic queue) ---
    if do_measure:
        # Shard CSVs
        shard_csvs = []
        log_paths = []
        for d in devices:
            shard_csvs.append(os.path.join(model_dir, f"{os.path.basename(model_dir)}Measured.gpu{d}.csv"))
            log_paths.append(os.path.join(model_dir, "logs", f"measure_gpu{d}.log") if write_logs else None)

        # Build task list: base once + all bpws
        tasks = Queue()
        tasks_list = ["base"] + bpws

        # Enqueue only tasks not already measured in ANY shard? Keep it simple:
        # each worker skips done based on its own shard CSV.
        # We still enqueue everything; skip-done makes it cheap.
        for t in tasks_list:
            tasks.put(t)

        # Termination sentinels
        for _ in devices:
            tasks.put(None)

        procs: List[Process] = []
        for d, csvp, logp in zip(devices, shard_csvs, log_paths):
            p = Process(target=_worker_measure, args=(model_dir, d, csvp, tasks, logp, exllamav3_root))
            p.daemon = False
            p.start()
            procs.append(p)

        for p in procs:
            p.join()

        # Merge into the canonical CSV name
        out_csv = default_csv_path(model_dir)
        _merge_csvs(out_csv, shard_csvs)
        print(f"✅ merged CSV: {out_csv}")

    # --- Stage 3: report (stub for now) ---
    if do_report:
        print("🟡 report stage not implemented yet")

    # --- Stage 4: cleanup (later; keep as stub) ---
    if cleanup:
        print("🟡 cleanup stage not implemented yet")

    return 0
