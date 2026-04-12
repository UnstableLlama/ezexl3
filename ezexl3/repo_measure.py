from __future__ import annotations

import os
import re
import sys
import threading
import time
from multiprocessing import Process, Queue
from typing import Any, Callable, Dict, List, Optional, Tuple

from ezexl3.measure import default_csv_path, file_size_gib
from ezexl3.measure_db import default_db_path, export_csv, migrate_csv_to_db, read_all_rows as _read_db_rows, upsert_row
from ezexl3.repo_progress import (
    _cleanup_gpu_progress,
    _clear_and_redraw_progress,
    _init_gpu_progress,
    _print_above_progress,
    _print_msg_with_progress,
    _redraw_gpu_progress,
)
from ezexl3.repo_subprocess import _run_catbench_subprocess, _run_measure_subprocess

_KL_RE = re.compile(
    r"(?:KL|K/L)\s+divergence(?:\s+\(A,\s+B\))?:\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)",
    re.IGNORECASE,
)
_PPL_RE = re.compile(
    r"Perplexity:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)",
    re.IGNORECASE,
)


def _bpw_sort_key(w: str):
    if w == "bf16":
        return (-1.0, w)
    try:
        return (float(w), w)
    except Exception:
        return (1e9, w)


def _task_to_csv_label(task: str) -> str:
    return "bf16" if task == "base" else task


def _filter_measure_tasks_for_checkpoint(requested_tasks: List[str], existing_labels: set[str]) -> List[str]:
    return [task for task in requested_tasks if _task_to_csv_label(task) not in existing_labels]


def _parse_measure_args(measure_args: List[str], default_devices: List[int]) -> tuple[int, List[int]]:
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


def _maybe_update_graph(model_dir: str, csv_path: str) -> None:
    import numpy as np
    from ezexl3.graph_svg import load_series, make_plot

    try:
        bpw, kld, ppl, gib, _ = load_series(csv_path, drop_bf16=True)
    except Exception:
        return

    valid = ~(np.isnan(kld) | np.isnan(ppl) | np.isnan(gib))
    bpw, kld, ppl, gib = bpw[valid], kld[valid], ppl[valid], gib[valid]

    if len(bpw) < 2:
        return
    basename = os.path.basename(os.path.abspath(model_dir)).lower()
    svg_path = os.path.join(model_dir, f"{basename}.svg")
    make_plot(bpw, kld, ppl, gib, title=basename, outfile=svg_path, add_checks=False)


def _init_measure_db(
    model_dir: str,
    devices: List[int],
    default_csv_path_fn: Callable = default_csv_path,
    default_db_path_fn: Callable = default_db_path,
    migrate_csv_to_db_fn: Callable = migrate_csv_to_db,
) -> Tuple[str, str]:
    out_csv = default_csv_path_fn(model_dir)
    db_path = default_db_path_fn(model_dir)
    if os.path.exists(out_csv):
        migrate_csv_to_db_fn(out_csv, db_path)
    for d in devices:
        legacy_shard = os.path.join(model_dir, f"{os.path.basename(model_dir)}Measured.gpu{d}.csv")
        if os.path.exists(legacy_shard):
            migrate_csv_to_db_fn(legacy_shard, db_path)
    return db_path, out_csv


def _worker_measure(
    base_dir: str,
    device: int,
    db_path: str,
    tasks,
    results,
    log_path: Optional[str],
    ppl_rows: int = 100,
    run_measure_subprocess_fn: Callable = _run_measure_subprocess,
    run_catbench_subprocess_fn: Callable = _run_catbench_subprocess,
    model_diff_script: Optional[str] = None,
    file_size_gib_fn: Callable = file_size_gib,
    upsert_row_fn: Callable = upsert_row,
    executable: str = sys.executable,
    catbench_cache_tokens: int = 4608,
) -> None:
    log_f = None
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_f = open(log_path, "w")

    while True:
        job = tasks.get()
        if job is None:
            results.put(None)
            break

        task_label = job["label"]
        phase = job["phase"]
        label = "bf16" if task_label == "base" else str(task_label)
        model_dir = base_dir if task_label == "base" else os.path.join(base_dir, str(task_label))
        phase_tag = phase.upper()
        results.put({"event": "start", "device": device, "label": label, "phase": phase})

        try:
            if phase == "kl":
                kl_cmd = [
                    executable,
                    model_diff_script,
                    "-ma", base_dir,
                    "-mb", model_dir,
                    "-r", "10",
                    "-d", str(device),
                ]
                kl_out = run_measure_subprocess_fn(kl_cmd, device, results, f"{label} KL", log_f)
                kl_match = _KL_RE.search(kl_out)
                if not kl_match:
                    raise ValueError("Could not parse model_diff output (KL Divergence pattern did not match).")
                kl_div = float(kl_match.group(1))

                gib = file_size_gib_fn(model_dir)
                upsert_row_fn(db_path, weights=label, kl_div=str(kl_div), gib=str(gib))
                row = {"weights": label, "KL Div": kl_div, "PPL r-100": "", "GiB": gib}
                results.put({"event": "done", "device": device, "label": label, "phase": phase, "row": row})

            elif phase == "ppl":
                ppl_cmd = [
                    executable,
                    "-m", "ezexl3.ppl_layer",
                    "-m", model_dir,
                    "-r", str(ppl_rows),
                    "-d", str(device),
                ]
                ppl_out = run_measure_subprocess_fn(ppl_cmd, device, results, f"{label} PPL", log_f)
                ppl_match = _PPL_RE.search(ppl_out)
                if not ppl_match:
                    raise ValueError("Could not parse ppl_layer output (Perplexity pattern did not match).")
                ppl = float(ppl_match.group(1))

                gib = file_size_gib_fn(model_dir)
                if task_label == "base":
                    upsert_row_fn(db_path, weights=label, kl_div="0.0", ppl=str(ppl), gib=str(gib))
                    row = {"weights": label, "KL Div": 0.0, "PPL r-100": ppl, "GiB": gib}
                else:
                    upsert_row_fn(db_path, weights=label, ppl=str(ppl), gib=str(gib))
                    row = {"weights": label, "KL Div": "", "PPL r-100": ppl, "GiB": gib}
                results.put({"event": "done", "device": device, "label": label, "phase": phase, "row": row})

            elif phase == "catbench":
                n_samples = job.get("n_samples", 3)
                catbench_out_dir = os.path.join(base_dir, "catbench")
                catbench_cmd = [
                    executable,
                    "-m", "ezexl3.catbench",
                    "-m", model_dir,
                    "-cs", str(catbench_cache_tokens),
                    "-n", str(n_samples),
                    "-o", catbench_out_dir,
                    "-l", label,
                ]
                run_catbench_subprocess_fn(catbench_cmd, device, results, f"{label} CAT", log_f)
                results.put({"event": "done", "device": device, "label": label, "phase": phase, "row": {}})

            else:
                # --- Eval scripts (diversity, humaneval, ifbench, ...) ---
                from ezexl3.evals import (
                    EVAL_REGISTRY,
                    RESULT_EXTRACTORS,
                    build_eval_cmd,
                    run_eval_subprocess,
                )
                if phase in EVAL_REGISTRY:
                    eval_def = EVAL_REGISTRY[phase]
                    eval_arg = job.get("eval_arg", 0)
                    eval_cmd = build_eval_cmd(
                        phase, model_dir, device, base_dir, label, eval_arg,
                    )
                    eval_out = run_eval_subprocess(
                        eval_cmd, device, results,
                        f"{label} {eval_def.phase_label}",
                        phase, log_f,
                        cuda_visible_devices=str(device),
                    )
                    extractor = RESULT_EXTRACTORS[phase]
                    result_dict = extractor(eval_out)
                    upsert_row_fn(db_path, weights=label, **result_dict)

                    # For perf, also write the detailed per-context-length
                    # curve to the dedicated perf database.
                    if phase == "perf":
                        try:
                            from ezexl3.evals import extract_perf_detail
                            from ezexl3.perf_db import (
                                default_perf_db_path,
                                upsert_perf_results,
                            )
                            detail = extract_perf_detail(eval_out)
                            if detail["prefill"] or detail["generation"]:
                                perf_db = default_perf_db_path(base_dir)
                                upsert_perf_results(
                                    perf_db, label,
                                    detail["prefill"],
                                    detail["generation"],
                                )
                        except Exception:
                            pass  # non-fatal: summary already saved

                    results.put({
                        "event": "done", "device": device, "label": label,
                        "phase": phase, "row": result_dict,
                    })

        except Exception as e:
            import traceback
            if log_f:
                traceback.print_exc(file=log_f)
                log_f.flush()
            results.put({"event": "error", "device": device, "label": label, "phase": phase, "error": str(e)})

    if log_f:
        log_f.close()


def run_measure_single_bpw(
    model_dir: str,
    bpw: str,
    devices: List[int],
    db_path: str,
    ppl_rows: int = 100,
    write_logs: bool = True,
    include_base_ppl: bool = False,
    read_db_rows_fn: Callable = _read_db_rows,
    task_to_csv_label_fn: Callable = _task_to_csv_label,
    process_cls=Process,
    queue_cls=Queue,
    worker_measure_fn: Callable = _worker_measure,
    init_gpu_progress_fn: Callable = _init_gpu_progress,
    redraw_gpu_progress_fn: Callable = _redraw_gpu_progress,
    print_msg_with_progress_fn: Callable = _print_msg_with_progress,
    cleanup_gpu_progress_fn: Callable = _cleanup_gpu_progress,
    export_csv_fn: Callable = export_csv,
    maybe_update_graph_fn: Callable = _maybe_update_graph,
    default_csv_path_fn: Callable = default_csv_path,
    sleep_fn: Callable = time.sleep,
) -> int:
    label = task_to_csv_label_fn(bpw)
    existing_rows = read_db_rows_fn(db_path)

    all_tasks: List[dict] = []
    if include_base_ppl:
        base_row = existing_rows.get("bf16", {})
        if not (base_row.get("PPL r-100") or "").strip():
            all_tasks.append({"label": "base", "phase": "ppl"})

    row = existing_rows.get(label, {})
    has_kl = bool((row.get("KL Div") or "").strip())
    has_ppl = bool((row.get("PPL r-100") or "").strip())

    if bpw != "base" and not has_kl:
        all_tasks.append({"label": bpw, "phase": "kl"})
    if not has_ppl:
        all_tasks.append({"label": bpw, "phase": "ppl"})

    if not all_tasks:
        print(f"  🟦 {label}: already measured, skipping")
        return 0

    n_workers = min(len(devices), len(all_tasks))
    worker_devices = devices[:n_workers]

    tasks_q = queue_cls()
    results_q = queue_cls()

    for t in all_tasks:
        tasks_q.put(t)
    for _ in worker_devices:
        tasks_q.put(None)

    log_paths = []
    for d in worker_devices:
        if write_logs:
            log_paths.append(os.path.join(model_dir, "logs", f"measure_gpu{d}_bpw{label}.log"))
        else:
            log_paths.append(None)

    procs = []
    for d, logp in zip(worker_devices, log_paths):
        p = process_cls(target=worker_measure_fn, args=(model_dir, d, db_path, tasks_q, results_q, logp, ppl_rows))
        p.daemon = False
        p.start()
        procs.append(p)
        if len(worker_devices) > 1:
            sleep_fn(2.0)

    task_descs = [f"{task_to_csv_label_fn(t['label'])} {t['phase'].upper()}" for t in all_tasks]
    print(f"  📊 Measuring {label}: {', '.join(task_descs)} on {n_workers} GPU(s)...")

    use_ansi = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    gpu_status: Dict[int, str] = {d: "idle" for d in worker_devices}
    num_lines = n_workers
    init_gpu_progress_fn(use_ansi, gpu_status)

    active_workers = n_workers
    failures = 0
    while active_workers > 0:
        res = results_q.get()
        if res is None:
            active_workers -= 1
            continue

        event = res["event"]
        if event == "progress":
            gpu = res["device"]
            gpu_status[gpu] = res["text"]
            redraw_gpu_progress_fn(use_ansi, gpu_status, num_lines)
            continue

        res_label = res.get("label", "")
        phase = res.get("phase", "")
        gpu = res.get("device", "?")
        phase_tag = phase.upper()

        if event == "start":
            msg = f"🧪 [GPU {gpu}] START {res_label} {phase_tag}"
            gpu_status[gpu] = f"{res_label} {phase_tag} | starting..."
        elif event == "done":
            row = res["row"]
            if phase == "kl":
                msg = f"✅ [GPU {gpu}] DONE {res_label} KL: KL={row.get('KL Div', 'N/A')}"
            else:
                msg = f"✅ [GPU {gpu}] DONE {res_label} PPL: PPL={row.get('PPL r-100', 'N/A')}"
            gpu_status[gpu] = "idle"
            try:
                out_csv = default_csv_path_fn(model_dir)
                export_csv_fn(db_path, out_csv)
                maybe_update_graph_fn(model_dir, out_csv)
            except Exception:
                pass
        elif event == "error":
            failures += 1
            msg = f"🔴 [GPU {gpu}] FAIL {res_label} {phase_tag}: {res['error']}"
            gpu_status[gpu] = "idle"
        else:
            continue

        print_msg_with_progress_fn(msg, use_ansi, gpu_status, num_lines)

    cleanup_gpu_progress_fn(use_ansi, num_lines)

    for p in procs:
        p.join()

    if failures:
        print(f"  ❌ Measurement failed for {label} with {failures} error(s)")
        return 1
    return 0


def run_measure_stage(
    model_dir: str,
    bpws: List[str],
    devices: List[int],
    write_logs: bool = True,
    measure_args: Optional[List[str]] = None,
    catbench_n: int = 0,
    evals: Optional[Dict[str, Any]] = None,
    skip_kl: bool = False,
    skip_ppl: bool = False,
    parse_measure_args_fn: Callable = _parse_measure_args,
    init_measure_db_fn: Callable = _init_measure_db,
    read_db_rows_fn: Callable = _read_db_rows,
    task_to_csv_label_fn: Callable = _task_to_csv_label,
    catbench_file_prefix_fn: Optional[Callable] = None,
    catbench_has_output_fn: Optional[Callable] = None,
    catbench_generate_svgs_fn: Optional[Callable] = None,
    file_size_gib_fn: Callable = file_size_gib,
    upsert_row_fn: Callable = upsert_row,
    process_cls=Process,
    queue_cls=Queue,
    worker_measure_fn: Callable = _worker_measure,
    init_gpu_progress_fn: Callable = _init_gpu_progress,
    redraw_gpu_progress_fn: Callable = _redraw_gpu_progress,
    print_msg_with_progress_fn: Callable = _print_msg_with_progress,
    cleanup_gpu_progress_fn: Callable = _cleanup_gpu_progress,
    clear_and_redraw_progress_fn: Callable = _clear_and_redraw_progress,
    print_above_progress_fn: Callable = _print_above_progress,
    export_csv_fn: Callable = export_csv,
    maybe_update_graph_fn: Callable = _maybe_update_graph,
    run_catbench_subprocess_fn: Callable = _run_catbench_subprocess,
    catbench_cache_tokens: int = 4608,
    executable: str = sys.executable,
    sleep_fn: Callable = time.sleep,
) -> int:
    model_dir = os.path.abspath(model_dir)
    bpws = [str(b) for b in bpws]
    devices = list(devices)
    ppl_rows, devices = parse_measure_args_fn(measure_args or [], devices)
    if not devices:
        raise ValueError("No CUDA devices available for measure stage. Provide -d/--devices.")

    db_path, out_csv = init_measure_db_fn(model_dir, devices)

    log_paths = [os.path.join(model_dir, "logs", f"measure_gpu{d}.log") if write_logs else None for d in devices]
    existing_rows = read_db_rows_fn(db_path)

    gib_filled = []
    all_labels_to_check = [task_to_csv_label_fn(b) for b in bpws]
    if "bf16" not in all_labels_to_check:
        all_labels_to_check.append("bf16")
    for lbl in all_labels_to_check:
        row = existing_rows.get(lbl, {})
        if not bool((row.get("GiB") or "").strip()):
            quant_dir = model_dir if lbl == "bf16" else os.path.join(model_dir, lbl)
            gib = file_size_gib_fn(quant_dir)
            if gib > 0:
                upsert_row_fn(db_path, weights=lbl, gib=str(round(gib, 2)))
                gib_filled.append(lbl)

    print("\n============================================================")
    print("📊 Measurement Phase")
    print("============================================================")
    if gib_filled:
        print(f"📏 Filled GiB for: {', '.join(gib_filled)}")

    kl_tasks: List[dict] = []
    ppl_tasks: List[dict] = []
    skipped_kl: List[str] = []
    skipped_ppl: List[str] = []

    for bpw in bpws:
        label = task_to_csv_label_fn(bpw)
        row = existing_rows.get(label, {})
        has_kl = bool((row.get("KL Div") or "").strip())
        has_ppl = bool((row.get("PPL r-100") or "").strip())
        if bpw != "base" and not has_kl and not skip_kl:
            kl_tasks.append({"label": bpw, "phase": "kl"})
        elif bpw != "base":
            skipped_kl.append(label)
        if not has_ppl and not skip_ppl:
            ppl_tasks.append({"label": bpw, "phase": "ppl"})
        else:
            skipped_ppl.append(label)

    base_row = existing_rows.get("bf16", {})
    if not bool((base_row.get("PPL r-100") or "").strip()) and not skip_ppl:
        if not any(t["label"] == "base" for t in ppl_tasks):
            ppl_tasks.append({"label": "base", "phase": "ppl"})
    else:
        if "bf16" not in skipped_ppl:
            skipped_ppl.append("bf16")

    if skipped_kl:
        print(f"🟦 skipping KL divergence: {', '.join(skipped_kl)} (already measured)")
    if skipped_ppl:
        print(f"🟦 skipping perplexity: {', '.join(skipped_ppl)} (already measured)")

    catbench_tasks: List[dict] = []
    multi_gpu_catbench_tasks: List[dict] = []
    skipped_catbench: List[str] = []

    if catbench_n > 0:
        catbench_out_dir = os.path.join(model_dir, "catbench")
        for bpw in bpws:
            label = task_to_csv_label_fn(bpw)
            file_prefix = catbench_file_prefix_fn(label)
            if not catbench_has_output_fn(catbench_out_dir, file_prefix, catbench_n):
                catbench_tasks.append({"label": bpw, "phase": "catbench", "n_samples": catbench_n})
            else:
                skipped_catbench.append(label)
        if not catbench_has_output_fn(catbench_out_dir, "bf16", catbench_n):
            catbench_tasks.append({"label": "base", "phase": "catbench", "n_samples": catbench_n})
        else:
            skipped_catbench.append("bf16")
        if skipped_catbench:
            print(f"🟦 skipping catbench: {', '.join(skipped_catbench)} (txt samples exist)")

        if len(devices) > 1 and catbench_tasks:
            from ezexl3.catbench import check_multi_gpu_fit, check_vram_fit

            single_gpu = []
            for task in catbench_tasks:
                task_label = task["label"]
                task_model_dir = model_dir if task_label == "base" else os.path.join(model_dir, str(task_label))
                fits, model_gib, avail_gib = check_vram_fit(task_model_dir, devices[0])
                if fits:
                    single_gpu.append(task)
                else:
                    multi_fits, model_gib, total_avail = check_multi_gpu_fit(task_model_dir, devices)
                    if multi_fits:
                        task["device_str"] = ",".join(str(d) for d in devices)
                        multi_gpu_catbench_tasks.append(task)
                    else:
                        task_disp = "bf16" if task_label == "base" else str(task_label)
                        print(f"  ⚠️  Skipping catbench for {task_disp}: {model_gib:.1f} GiB model won't fit ({total_avail:.1f} GiB available across {len(devices)} GPUs)")
            catbench_tasks = single_gpu

    # --- Eval tasks ---
    eval_tasks: List[dict] = []
    skipped_eval: List[str] = []
    enabled_evals = evals or {}

    if enabled_evals:
        from ezexl3.evals import EVAL_QUEUE_ORDER, EVAL_REGISTRY, eval_has_result

        for eval_name in EVAL_QUEUE_ORDER:
            if eval_name not in enabled_evals:
                continue
            eval_arg = enabled_evals[eval_name]
            all_targets = list(bpws)
            # Include base/bf16 for all evals
            if "base" not in all_targets:
                all_targets.append("base")
            for bpw in all_targets:
                label = task_to_csv_label_fn(bpw)
                if eval_has_result(db_path, label, eval_name):
                    skipped_eval.append(f"{label}/{eval_name}")
                else:
                    eval_tasks.append({
                        "label": bpw, "phase": eval_name, "eval_arg": eval_arg,
                    })

        if skipped_eval:
            print(f"🟦 skipping evals: {', '.join(skipped_eval)} (already measured)")

    total_jobs = (len(kl_tasks) + len(ppl_tasks) + len(catbench_tasks)
                  + len(multi_gpu_catbench_tasks) + len(eval_tasks))
    if total_jobs == 0:
        if catbench_n > 0:
            catbench_out_dir = os.path.join(model_dir, "catbench")
            print("🎨 Generating SVGs from catbench results...")
            n_svgs = catbench_generate_svgs_fn(catbench_out_dir)
            print(f"✅ {n_svgs} SVGs generated.")
        else:
            print("✅ All requested measurement phases already exist. Nothing to do.")
        return 0

    n_kl = len(kl_tasks)
    n_ppl = len(ppl_tasks)
    n_cat = len(catbench_tasks) + len(multi_gpu_catbench_tasks)
    n_eval = len(eval_tasks)

    if multi_gpu_catbench_tasks:
        from queue import Queue as TQueue

        mgpu_use_ansi = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        print(f"\n🐱 Running {len(multi_gpu_catbench_tasks)} multi-GPU catbench job(s)...")
        mgpu_status: Dict[int, str] = {d: "idle" for d in devices}
        mgpu_num_lines = len(devices)
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
                executable, "-m", "ezexl3.catbench",
                "-m", task_model_dir,
                "-gs", ",".join("99" for _ in device_str.split(",")),
                "-cs", str(catbench_cache_tokens),
                "-n", str(task.get("n_samples", 3)),
                "-o", os.path.join(model_dir, "catbench"),
                "-l", label,
            ]
            phase_label = f"{label} CAT"
            mgpu_results: TQueue = TQueue()
            mgpu_error: List[Optional[Exception]] = [None]

            def _run_mgpu_catbench(_cmd=catbench_cmd, _dev=devices[0], _q=mgpu_results, _pl=phase_label, _cvd=device_str):
                try:
                    run_catbench_subprocess_fn(_cmd, _dev, _q, _pl, cuda_visible_devices=_cvd)
                except Exception as exc:
                    mgpu_error[0] = exc
                _q.put(None)

            t = threading.Thread(target=_run_mgpu_catbench)
            t.start()

            while True:
                ev = mgpu_results.get()
                if ev is None:
                    break
                if ev["event"] == "progress":
                    for d in devices:
                        mgpu_status[d] = ev["text"]
                    if mgpu_use_ansi:
                        clear_and_redraw_progress_fn(mgpu_status, mgpu_num_lines)

            t.join()

            if mgpu_error[0] is not None:
                msg = f"🔴 Multi-GPU catbench failed for {label}: {mgpu_error[0]}"
            else:
                msg = f"🐱 DONE {label} CATBENCH (multi-GPU [{device_str}])"

            for d in devices:
                mgpu_status[d] = "idle"
            if mgpu_use_ansi:
                print_above_progress_fn(msg, mgpu_status, mgpu_num_lines)
            else:
                print(msg)

        if mgpu_use_ansi:
            sys.stdout.write(f"\033[{mgpu_num_lines}A")
            for _ in range(mgpu_num_lines):
                sys.stdout.write("\033[2K\n")
            sys.stdout.write(f"\033[{mgpu_num_lines}A")
            sys.stdout.flush()

    remaining_jobs = len(kl_tasks) + len(ppl_tasks) + len(catbench_tasks) + len(eval_tasks)
    if remaining_jobs == 0:
        if multi_gpu_catbench_tasks:
            catbench_out_dir = os.path.join(model_dir, "catbench")
            print("\n🎨 Generating SVGs from catbench results...")
            n_svgs = catbench_generate_svgs_fn(catbench_out_dir)
            print(f"✅ All catbench jobs complete: {n_svgs} SVGs generated.")
        return 0

    tasks = queue_cls()
    results = queue_cls()
    bpw_tasks: Dict[str, List[dict]] = {}
    for t in ppl_tasks:
        bpw_tasks.setdefault(t["label"], []).append(t)
    for t in kl_tasks:
        bpw_tasks.setdefault(t["label"], []).append(t)

    def _local_bpw_sort_key(label: str) -> float:
        if label == "base":
            return -1.0
        try:
            return float(label)
        except ValueError:
            return 999.0

    for label in sorted(bpw_tasks, key=_local_bpw_sort_key):
        for t in bpw_tasks[label]:
            tasks.put(t)
    for t in catbench_tasks:
        tasks.put(t)
    for t in eval_tasks:
        tasks.put(t)
    for _ in devices:
        tasks.put(None)

    procs = []
    for d, logp in zip(devices, log_paths):
        p = process_cls(target=worker_measure_fn, args=(model_dir, d, db_path, tasks, results, logp, ppl_rows))
        p.daemon = False
        p.start()
        procs.append(p)
        sleep_fn(2.0)

    cat_msg = f" + {n_cat} CAT" if n_cat else ""
    eval_msg = f" + {n_eval} EVAL" if n_eval else ""
    print(f"\n🚀 Measuring {n_kl} KL + {n_ppl} PPL{cat_msg}{eval_msg} jobs on {len(devices)} GPUs...")

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

        label = res["label"]
        phase = res.get("phase", "")
        phase_tag = phase.upper()

        if event == "start":
            msg = f"🧪 [GPU {gpu}] START {label} {phase_tag}"
            gpu_status[gpu] = f"{label} {phase_tag} | starting..."
        elif event == "done":
            row = res["row"]
            if phase == "kl":
                msg = f"✅ [GPU {gpu}] DONE {label} KL: KL={row.get('KL Div', 'N/A')}"
            elif phase == "ppl":
                msg = f"✅ [GPU {gpu}] DONE {label} PPL: PPL={row.get('PPL r-100', 'N/A')}"
            elif phase == "catbench":
                msg = f"🐱 [GPU {gpu}] DONE {label} CATBENCH"
            else:
                from ezexl3.evals import (
                    EVAL_REGISTRY,
                    format_eval_result,
                    result_is_empty,
                )
                eval_def = EVAL_REGISTRY.get(phase)
                if eval_def is not None:
                    summary = format_eval_result(phase, row) or "N/A"
                    if result_is_empty(phase, row):
                        msg = (
                            f"⚠️  [GPU {gpu}] DONE {label} {eval_def.phase_label}: "
                            f"no results extracted (subprocess finished but regex did not match)"
                        )
                        failures += 1
                    else:
                        msg = f"✅ [GPU {gpu}] DONE {label} {eval_def.phase_label}: {summary}"
                else:
                    msg = f"✅ [GPU {gpu}] DONE {label} {phase_tag}"
            gpu_status[gpu] = "idle"
            # Flush CSV after every DB-writing phase so the on-disk snapshot
            # tracks reality. Graph update stays gated to KL/PPL because evals
            # don't feed the graph.
            if phase in ("kl", "ppl"):
                try:
                    export_csv_fn(db_path, out_csv)
                    maybe_update_graph_fn(model_dir, out_csv)
                except Exception:
                    pass
            elif phase != "catbench":
                try:
                    export_csv_fn(db_path, out_csv)
                except Exception:
                    pass
        elif event == "error":
            failures += 1
            msg = f"🔴 [GPU {gpu}] FAIL {label} {phase_tag}: {res['error']}"
            gpu_status[gpu] = "idle"
        else:
            continue

        print_msg_with_progress_fn(msg, use_ansi, gpu_status, num_lines)

    cleanup_gpu_progress_fn(use_ansi, num_lines)

    for p in procs:
        p.join()

    export_csv_fn(db_path, out_csv)

    if catbench_n > 0:
        catbench_out_dir = os.path.join(model_dir, "catbench")
        print("\n🎨 Generating SVGs from catbench results...")
        n_svgs = catbench_generate_svgs_fn(catbench_out_dir)
        print(f"✅ Catbench: {n_svgs} SVGs generated.")

    if failures:
        print(f"⚠️ Measurement stage completed with {failures} failure(s). Merged CSV: {out_csv}")
        return 1
    print(f"✅ All measurements complete. Merged CSV: {out_csv}")
    return 0
