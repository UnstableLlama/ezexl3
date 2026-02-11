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
    fieldnames = ["weights", "K/L Div", "PPL r-100", "GiB"]
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
    results: "Queue[Optional[dict]]",
    log_path: Optional[str],
    ppl_rows: int = 100,
) -> None:
    import traceback

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
            results.put(None)  # Sentinel
            break

        # Each item is a single quant label ("base" or "2" etc.)
        try:
            # We want to capture the result row. 
            # run_measure currently doesn't return it, so we'll read the last row of the shard CSV 
            # after it runs, or modify run_measure.
            # Let's modify run_measure in measure.py to return the row.
            
            from ezexl3.measure import run_measure
            row = run_measure(
                base_dir=base_dir,
                quants=[item],
                device=device,
                csv_path=csv_path,
                skip_done=True,
                return_row=True,
                ppl_rows=ppl_rows,
            )
            if row:
                results.put(row)
        except Exception as e:
            print(f"🔴 ERROR measuring '{item}': {e}")
            traceback.print_exc()
            # Send an error placeholder?
            results.put({"weights": item if item != "base" else "bf16", "error": str(e)})

    if log_path:
        sys.stdout.flush()
        sys.stderr.flush()
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
) -> int:
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
) -> int:
    model_dir = os.path.abspath(model_dir)
    bpws = [str(b) for b in bpws]
    devices = list(devices)
    ppl_rows, devices = _parse_measure_args(measure_args or [], devices)

    # Shard CSVs
    shard_csvs = []
    log_paths = []
    for d in devices:
        shard_csvs.append(os.path.join(model_dir, f"{os.path.basename(model_dir)}Measured.gpu{d}.csv"))
        log_paths.append(os.path.join(model_dir, "logs", f"measure_gpu{d}.log") if write_logs else None)

    # Build task list: base once + all bpws
    tasks = Queue()
    results = Queue()
    tasks_list = bpws + ["base"]

    for t in tasks_list:
        tasks.put(t)

    # Termination sentinels
    for _ in devices:
        tasks.put(None)

    procs: List[Process] = []
    for d, csvp, logp in zip(devices, shard_csvs, log_paths):
        p = Process(target=_worker_measure, args=(model_dir, d, csvp, tasks, results, logp, ppl_rows))
        p.daemon = False
        p.start()
        procs.append(p)
        time.sleep(2.0)

    # Result listener loop
    out_csv = default_csv_path(model_dir)
    active_workers = len(devices)
    all_results = {}

    # Seed merged output from any existing shard state.
    # This is important for resume flows where workers may skip already-done rows
    # and therefore not emit fresh result events for every quant.
    _merge_csvs(out_csv, shard_csvs)

    print(f"\n🚀 Measuring {len(tasks_list)} items on {len(devices)} GPUs...")
    
    while active_workers > 0:
        res = results.get()
        if res is None:
            active_workers -= 1
            continue
        
        w = res.get("weights")
        if not w: continue
        
        if "error" in res:
            print(f"🔴 {w}: FAILED - {res['error']}")
        else:
            print(f"✅ {w}: PPL(r100)={res.get('PPL r-100', 'N/A')}")
        
        all_results[w] = res
        
        # Merge partially to keep the main CSV up to date
        _merge_csvs(out_csv, shard_csvs)

    for p in procs:
        p.join()

    # Always do one final merge after workers exit to ensure every shard write is
    # reflected in the main CSV, even when a worker skipped tasks (no result event)
    # or when the last merge happened before another worker finished flushing rows.
    _merge_csvs(out_csv, shard_csvs)

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
) -> int:
    # --- Stage 1: quantize ---
    if do_quant:
        rc = run_quant_stage(
            model_dir=model_dir,
            bpws=bpws,
            devices=devices,
            device_ratios=device_ratios,
            quant_args=quant_args,
        )
        if rc != 0:
            return rc

    # --- Stage 2: measure (sharded, dynamic queue) ---
    if do_measure:
        rc = run_measure_stage(
            model_dir=model_dir,
            bpws=bpws,
            devices=devices,
            write_logs=write_logs,
            measure_args=measure_args,
        )
        if rc != 0:
            return rc

    # --- Stage 3: README generation ---
    if do_readme:
        from ezexl3.readme import run_readme
        print("Generating README...")
        run_readme(model_dir, template_name=template, interactive=interactive)

    # --- Stage 4: cleanup ---
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
