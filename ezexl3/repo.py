# ezexl3/repo.py
from __future__ import annotations

import csv
import importlib.util
import math
import os
import subprocess
import sys
import time
from multiprocessing import Process, Queue
from typing import Dict, List, Optional, Tuple

from ezexl3.quantize import run as quant_run
from ezexl3.measure import default_csv_path


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


def _split_integer_fractional_bpws(bpws: List[str]) -> Tuple[List[str], List[str]]:
    integer_bpws: List[str] = []
    fractional_bpws: List[str] = []

    for raw in bpws:
        normalized = _normalize_bpw_str(raw)
        value = float(normalized)
        if math.isclose(value, round(value), abs_tol=1e-9):
            integer_bpws.append(str(int(round(value))))
        else:
            fractional_bpws.append(normalized)
    return integer_bpws, fractional_bpws


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
    ints, fracs = _split_integer_fractional_bpws(bpws)
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
        "requested_fractionals": requested_fracs,
        "quant_integer_queue": quant_ints,
        "measure_queue": measure_targets,
    }


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


def _build_fractional_jobs(model_dir: str, fractional_bpws: List[str]) -> Tuple[List[dict], List[dict]]:
    measurements_dir = os.path.join(model_dir, "measurements")
    os.makedirs(measurements_dir, exist_ok=True)

    compare_jobs_by_pair: Dict[Tuple[str, str], dict] = {}
    optimize_jobs: List[dict] = []

    for frac in fractional_bpws:
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
                "fractional": frac,
                "out_dir": out_dir,
                "measure_json": measure_json,
                "low": low,
                "high": high,
            }
        )

    return list(compare_jobs_by_pair.values()), optimize_jobs


def _worker_fractional_compare(
    measure_script: str,
    model_dir: str,
    device: int,
    primary_device: int,
    tasks: "Queue[Optional[dict]]",
    results: "Queue[Optional[dict]]",
    log_path: Optional[str],
) -> None:
    import traceback

    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_f = open(log_path, "w")
        sys.stdout = log_f  # type: ignore
        sys.stderr = log_f  # type: ignore
    elif device != primary_device:
        devnull_f = open(os.devnull, "w")
        sys.stdout = devnull_f  # type: ignore
        sys.stderr = devnull_f  # type: ignore

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
                "3",
            ]
            _run_cmd(cmd)
            results.put({"event": "done", "device": device, "job": job, "label": label})
        except Exception as e:
            traceback.print_exc()
            results.put({"event": "error", "device": device, "job": job, "label": label, "error": str(e)})

    if log_path:
        sys.stdout.flush()
        sys.stderr.flush()
        log_f.close()
    elif device != primary_device:
        sys.stdout.flush()
        sys.stderr.flush()
        devnull_f.close()


def _run_fractional_compare_queue(
    model_dir: str,
    compare_jobs: List[dict],
    devices: List[int],
    measure_script: str,
    write_logs: bool = True,
) -> None:
    if not compare_jobs:
        return
    if not devices:
        raise ValueError("No CUDA devices available for fractional comparative measure stage")

    tasks: Queue = Queue()
    results: Queue = Queue()

    for job in compare_jobs:
        tasks.put(job)
    for _ in devices:
        tasks.put(None)

    procs: List[Process] = []
    primary_device = devices[0]
    for device in devices:
        # Keep primary GPU output in terminal; redirect all other GPUs to per-GPU logs
        # (or devnull when logs are disabled) to reduce console clutter.
        log_path = (
            os.path.join(model_dir, "logs", f"fractional_compare_gpu{device}.log")
            if write_logs and device != primary_device
            else None
        )
        p = Process(
            target=_worker_fractional_compare,
            args=(measure_script, model_dir, device, primary_device, tasks, results, log_path),
        )
        p.daemon = False
        p.start()
        procs.append(p)

    print(f"\n🚀 Fractional comparative measure: {len(compare_jobs)} jobs on {len(devices)} GPUs...")
    active_workers = len(devices)
    failures = 0
    while active_workers > 0:
        res = results.get()
        if res is None:
            active_workers -= 1
            continue
        job = res["job"]
        gpu = res["device"]
        label = f"{job['low']}-{job['high']}"
        targets = ",".join(job["targets"])
        if res["event"] == "start":
            print(f"🧪 [GPU {gpu}] START compare {label} for target(s): {targets}")
        elif res["event"] == "done":
            print(f"✅ [GPU {gpu}] DONE compare {label} for target(s): {targets} -> {job['measure_json']}")
        elif res["event"] == "error":
            failures += 1
            print(f"🔴 [GPU {gpu}] FAIL compare {label} for target(s): {targets} - {res['error']}")

    for p in procs:
        p.join()
    if failures:
        raise RuntimeError(f"Fractional comparative measure stage failed for {failures} job(s)")


def _run_fractional_opt_stage(
    model_dir: str,
    fractional_bpws: List[str],
    devices: List[int],
    write_logs: bool = True,
) -> None:
    if not fractional_bpws:
        return

    measure_script, optimize_script = _resolve_exllamav3_util_scripts()
    compare_jobs, optimize_jobs = _build_fractional_jobs(model_dir, fractional_bpws)

    queued_jobs: List[dict] = []
    for job in compare_jobs:
        label = f"{job['low']}-{job['high']}"
        if os.path.exists(job["measure_json"]):
            print(
                f"🟦 skipping comparative measure {label}: {os.path.basename(job['measure_json'])} already exists"
            )
            continue
        queued_jobs.append(job)

    _run_fractional_compare_queue(
        model_dir=model_dir,
        compare_jobs=queued_jobs,
        devices=devices,
        measure_script=measure_script,
        write_logs=write_logs,
    )

    for job in optimize_jobs:
        frac = job["fractional"]
        out_dir = job["out_dir"]
        if os.path.isdir(out_dir) and os.path.isfile(os.path.join(out_dir, "config.json")):
            print(f"🟦 skipping fractional optimize {frac}: output already exists")
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
        print(f"\n⚙️ Optimizing fractional quant {frac}")
        _run_cmd(optimize_cmd)


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
    fieldnames = ["weights", "KL Div", "PPL r-100", "GiB"]
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
    if not devices:
        raise ValueError("No CUDA devices available for measure stage. Provide -d/--devices.")

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
    include_graph: bool = True,
    include_measurements: bool = True,
) -> int:
    bpw_plan = _plan_repo_bpws(bpws)
    quant_bpws = bpw_plan["quant_integer_queue"]
    fractional_bpws = bpw_plan["requested_fractionals"]
    measure_bpws = bpw_plan["measure_queue"]

    auto_added = [b for b in quant_bpws if b not in bpw_plan["requested_integers"]]
    if auto_added:
        print(
            "ℹ️ Added required integer quants for fractional targets: "
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

    # --- Stage 2: fractional optimize ---
    if do_quant and fractional_bpws:
        _run_fractional_opt_stage(
            model_dir=model_dir,
            fractional_bpws=fractional_bpws,
            devices=devices,
            write_logs=write_logs,
        )

    # --- Stage 3: measure (sharded, dynamic queue) ---
    if do_measure:
        rc = run_measure_stage(
            model_dir=model_dir,
            bpws=measure_bpws,
            devices=devices,
            write_logs=write_logs,
            measure_args=measure_args,
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
