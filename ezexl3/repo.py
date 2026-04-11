# ezexl3/repo.py
from __future__ import annotations

import math
import os
import pty
import re
import select
import subprocess
import sys
import threading
import time
from multiprocessing import Process, Queue
from typing import Dict, IO, List, Optional, Tuple

from ezexl3.repo_plan import (
    _dedupe_preserve_order,
    _normalize_bpw_str,
    _plan_repo_bpws,
    _split_integer_optimized_bpws,
)
from ezexl3.repo_progress import (
    _build_synthetic_bar,
    _cleanup_gpu_progress,
    _clear_and_redraw_progress,
    _gpu_status_line,
    _init_gpu_progress,
    _print_above_progress,
    _print_msg_with_progress,
    _redraw_gpu_progress,
    _strip_ansi,
)
import ezexl3.repo_measure as repo_measure
import ezexl3.repo_optimized as repo_optimized
from ezexl3.repo_subprocess import (
    _run_catbench_subprocess,
    _run_cmd,
    _run_cmd_with_progress,
    _run_measure_subprocess,
)
from ezexl3.quantize import run as quant_run, run_one as quant_run_one
from ezexl3.measure import (
    _MODEL_DIFF_SCRIPT,
    default_csv_path,
    file_size_gib,
)
from ezexl3.measure_db import (
    default_db_path,
    export_csv,
    migrate_csv_to_db,
    read_all_rows as _read_db_rows,
    upsert_row,
)


_CATBENCH_CACHE_TOKENS = 4096 + 512  # prompt + generation headroom


def _catbench_file_prefix(label: str) -> str:
    """Convert a CSV label to a catbench SVG filename prefix."""
    if label in ("bf16", "base"):
        return "bf16"
    try:
        val = float(label)
        return f"{val:.2f}bpw"
    except (ValueError, TypeError):
        return label


def _catbench_has_output(catbench_dir: str, file_prefix: str, n_samples: int) -> bool:
    """Check if catbench has all N .txt samples for *file_prefix*.

    File naming convention:
      sample 1: {prefix}.txt  (canonical)
      sample 2: {prefix}_1.txt
      sample 3: {prefix}_2.txt
    """
    if not os.path.isdir(catbench_dir):
        return False
    count = 0
    for i in range(1, n_samples + 1):
        if i == 1:
            txt = os.path.join(catbench_dir, f"{file_prefix}.txt")
        else:
            txt = os.path.join(catbench_dir, f"{file_prefix}_{i - 1}.txt")
        if os.path.exists(txt):
            count += 1
    return count >= n_samples


def _catbench_generate_svgs(catbench_dir: str) -> int:
    """Batch-extract SVGs from all .txt files in catbench_dir.

    Groups .txt files by prefix, extracts SVGs, and names them
    sequentially based on successful extractions:
      first success  → {prefix}.svg
      second success → {prefix}_1.svg
      third success  → {prefix}_2.svg
      ...

    Any pre-existing .svg files for a prefix are removed first so
    numbering is always consistent.

    Returns total number of SVGs generated.
    """
    if not os.path.isdir(catbench_dir):
        return 0

    from ezexl3.catbench import extract_svg

    # Group txt files by prefix: "2.00bpw" → ["2.00bpw.txt", "2.00bpw_1.txt", ...]
    prefix_txts: Dict[str, List[str]] = {}
    for fn in sorted(os.listdir(catbench_dir)):
        if not fn.endswith(".txt"):
            continue
        # Match {prefix}.txt or {prefix}_{N}.txt
        m = re.match(r"^(.+?)(?:_\d+)?\.txt$", fn)
        if m:
            prefix = m.group(1)
            prefix_txts.setdefault(prefix, []).append(fn)

    total_svgs = 0
    for prefix, txt_files in sorted(prefix_txts.items()):
        # Remove any existing SVGs for this prefix to ensure clean numbering
        for fn in os.listdir(catbench_dir):
            if fn.endswith(".svg") and (fn == f"{prefix}.svg" or
                    re.match(rf"^{re.escape(prefix)}_\d+\.svg$", fn)):
                os.remove(os.path.join(catbench_dir, fn))

        svg_count = 0
        for txt_fn in txt_files:
            txt_path = os.path.join(catbench_dir, txt_fn)
            with open(txt_path, "r") as f:
                raw = f.read()

            svg_content = extract_svg(raw)
            if not svg_content:
                print(f"  ⚠️  No SVG extracted from {txt_fn}")
                continue

            # First successful SVG: {prefix}.svg, then {prefix}_1.svg, etc.
            if svg_count == 0:
                svg_fn = f"{prefix}.svg"
            else:
                svg_fn = f"{prefix}_{svg_count}.svg"

            svg_path = os.path.join(catbench_dir, svg_fn)
            with open(svg_path, "w") as f:
                f.write(svg_content)
            print(f"  🎨 {txt_fn} → {svg_fn} ({len(svg_content)} chars)")
            svg_count += 1

        total_svgs += svg_count

    return total_svgs


_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")
_MEASURE_SCRIPT = os.path.join(_VENDOR_DIR, "measure.py")
_OPTIMIZE_SCRIPT = os.path.join(_VENDOR_DIR, "optimize.py")


# ---------------------------------------------------------------------------
# ANSI progress-area rendering
# ---------------------------------------------------------------------------


def _build_optimized_jobs(model_dir: str, optimized_bpws: List[str]) -> Tuple[List[dict], List[dict]]:
    return repo_optimized._build_optimized_jobs(model_dir, optimized_bpws)


def _worker_optimized_compare(
    model_dir: str,
    device: int,
    layers: int,
    tasks: "Queue[Optional[dict]]",
    results: "Queue[Optional[dict]]",
    log_path: Optional[str],
) -> None:
    return repo_optimized._worker_optimized_compare(
        model_dir=model_dir,
        device=device,
        layers=layers,
        tasks=tasks,
        results=results,
        log_path=log_path,
        run_cmd_with_progress_fn=_run_cmd_with_progress,
        measure_script=_MEASURE_SCRIPT,
        executable=sys.executable,
    )


def _run_optimized_compare_queue(
    model_dir: str,
    compare_jobs: List[dict],
    devices: List[int],
    layers: int,
    write_logs: bool = True,
) -> None:
    return repo_optimized._run_optimized_compare_queue(
        model_dir=model_dir,
        compare_jobs=compare_jobs,
        devices=devices,
        layers=layers,
        write_logs=write_logs,
        process_cls=Process,
        queue_cls=Queue,
        worker_fn=_worker_optimized_compare,
        init_gpu_progress_fn=_init_gpu_progress,
        redraw_gpu_progress_fn=_redraw_gpu_progress,
        print_msg_with_progress_fn=_print_msg_with_progress,
        cleanup_gpu_progress_fn=_cleanup_gpu_progress,
    )


def _run_optimized_opt_stage(
    model_dir: str,
    optimized_bpws: List[str],
    devices: List[int],
    layers: int = 2,
    write_logs: bool = True,
) -> None:
    return repo_optimized._run_optimized_opt_stage(
        model_dir=model_dir,
        optimized_bpws=optimized_bpws,
        devices=devices,
        layers=layers,
        write_logs=write_logs,
        build_jobs_fn=_build_optimized_jobs,
        run_compare_queue_fn=_run_optimized_compare_queue,
        run_cmd_fn=_run_cmd,
        optimize_script=_OPTIMIZE_SCRIPT,
        executable=sys.executable,
    )
def _bpw_sort_key(w: str):
    return repo_measure._bpw_sort_key(w)


def _task_to_csv_label(task: str) -> str:
    return repo_measure._task_to_csv_label(task)


def _filter_measure_tasks_for_checkpoint(requested_tasks: List[str], existing_labels: set[str]) -> List[str]:
    return repo_measure._filter_measure_tasks_for_checkpoint(requested_tasks, existing_labels)


# ---------------------------------------------------------------------------
# Synthetic progress for measure subprocesses (ppl_layer / model_diff)
# ---------------------------------------------------------------------------


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


def _worker_measure(
    base_dir: str,
    device: int,
    db_path: str,
    tasks: "Queue[Optional[dict]]",
    results: "Queue[Optional[dict]]",
    log_path: Optional[str],
    ppl_rows: int = 100,
) -> None:
    return repo_measure._worker_measure(
        base_dir=base_dir,
        device=device,
        db_path=db_path,
        tasks=tasks,
        results=results,
        log_path=log_path,
        ppl_rows=ppl_rows,
        run_measure_subprocess_fn=_run_measure_subprocess,
        run_catbench_subprocess_fn=_run_catbench_subprocess,
        model_diff_script=_MODEL_DIFF_SCRIPT,
        file_size_gib_fn=file_size_gib,
        upsert_row_fn=upsert_row,
        executable=sys.executable,
        catbench_cache_tokens=_CATBENCH_CACHE_TOKENS,
    )


def _parse_measure_args(measure_args: List[str], default_devices: List[int]) -> tuple[int, List[int]]:
    return repo_measure._parse_measure_args(measure_args, default_devices)


def _maybe_update_graph(model_dir: str, csv_path: str) -> None:
    return repo_measure._maybe_update_graph(model_dir, csv_path)


def _init_measure_db(model_dir: str, devices: List[int]) -> Tuple[str, str]:
    return repo_measure._init_measure_db(
        model_dir=model_dir,
        devices=devices,
        default_csv_path_fn=default_csv_path,
        default_db_path_fn=default_db_path,
        migrate_csv_to_db_fn=migrate_csv_to_db,
    )


def _build_quant_forwarded(
    quant_args: List[str],
    devices: List[int],
    device_ratios: Optional[str],
) -> List[str]:
    """Build the forwarded arg list for quantize, injecting devices/ratios."""
    forwarded = list(quant_args)
    if devices and ("-d" not in forwarded and "--devices" not in forwarded):
        forwarded += ["-d", ",".join(str(d) for d in devices)]
    if device_ratios and ("-dr" not in forwarded and "--device-ratios" not in forwarded):
        forwarded += ["-dr", device_ratios]
    return forwarded


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

    forwarded = _build_quant_forwarded(quant_args, devices, device_ratios)

    rc = quant_run(
        models=[model_dir],
        bpws=bpws,
        forwarded=forwarded,
        out_template=out_template,
        w_template=w_template,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
    )
    return rc


def run_measure_single_bpw(
    model_dir: str,
    bpw: str,
    devices: List[int],
    db_path: str,
    ppl_rows: int = 100,
    write_logs: bool = True,
    include_base_ppl: bool = False,
) -> int:
    return repo_measure.run_measure_single_bpw(
        model_dir=model_dir,
        bpw=bpw,
        devices=devices,
        db_path=db_path,
        ppl_rows=ppl_rows,
        write_logs=write_logs,
        include_base_ppl=include_base_ppl,
        read_db_rows_fn=_read_db_rows,
        task_to_csv_label_fn=_task_to_csv_label,
        process_cls=Process,
        queue_cls=Queue,
        worker_measure_fn=_worker_measure,
        init_gpu_progress_fn=_init_gpu_progress,
        redraw_gpu_progress_fn=_redraw_gpu_progress,
        print_msg_with_progress_fn=_print_msg_with_progress,
        cleanup_gpu_progress_fn=_cleanup_gpu_progress,
        export_csv_fn=export_csv,
        maybe_update_graph_fn=_maybe_update_graph,
        default_csv_path_fn=default_csv_path,
        sleep_fn=time.sleep,
    )


def run_measure_stage(
    model_dir: str,
    bpws: List[str],
    devices: List[int],
    write_logs: bool = True,
    measure_args: Optional[List[str]] = None,
    catbench_n: int = 0,
) -> int:
    return repo_measure.run_measure_stage(
        model_dir=model_dir,
        bpws=bpws,
        devices=devices,
        write_logs=write_logs,
        measure_args=measure_args,
        catbench_n=catbench_n,
        parse_measure_args_fn=_parse_measure_args,
        init_measure_db_fn=_init_measure_db,
        read_db_rows_fn=_read_db_rows,
        task_to_csv_label_fn=_task_to_csv_label,
        catbench_file_prefix_fn=_catbench_file_prefix,
        catbench_has_output_fn=_catbench_has_output,
        catbench_generate_svgs_fn=_catbench_generate_svgs,
        file_size_gib_fn=file_size_gib,
        upsert_row_fn=upsert_row,
        process_cls=Process,
        queue_cls=Queue,
        worker_measure_fn=_worker_measure,
        init_gpu_progress_fn=_init_gpu_progress,
        redraw_gpu_progress_fn=_redraw_gpu_progress,
        print_msg_with_progress_fn=_print_msg_with_progress,
        cleanup_gpu_progress_fn=_cleanup_gpu_progress,
        clear_and_redraw_progress_fn=_clear_and_redraw_progress,
        print_above_progress_fn=_print_above_progress,
        export_csv_fn=export_csv,
        maybe_update_graph_fn=_maybe_update_graph,
        run_catbench_subprocess_fn=_run_catbench_subprocess,
        catbench_cache_tokens=_CATBENCH_CACHE_TOKENS,
        executable=sys.executable,
        sleep_fn=time.sleep,
    )


def run_repo(
    model_dir: str,
    bpws: List[str],
    devices: List[int],
    device_ratios: Optional[str],
    quant_args: List[str],
    measure_args: List[str],
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
    verify: bool = True,
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

    if verify and do_quant and do_measure:
        # --- INTERLEAVED MODE (default) ---
        # Quantize each integer BPW, then immediately verify KL+PPL
        # before moving to the next. Halts on any failure.
        model_dir = os.path.abspath(model_dir)
        forwarded = _build_quant_forwarded(quant_args, devices, device_ratios)
        ppl_rows, measure_devices = _parse_measure_args(measure_args or [], devices)
        db_path, _out_csv = _init_measure_db(model_dir, measure_devices)

        print("\n============================================================")
        print("🔁 Interleaved Quantize → Verify Pipeline")
        print(f"   {len(quant_bpws)} integer BPW(s), {len(optimized_bpws)} optimized BPW(s)")
        print(f"   {len(devices)} GPU(s) for quantization and measurement")
        print("============================================================")

        # Stage 1: quantize + verify each integer BPW
        for i, bpw in enumerate(quant_bpws):
            print(f"\n--- [{i+1}/{len(quant_bpws)}] BPW {bpw} ---")

            ok = quant_run_one(
                model_dir, str(bpw), forwarded,
                out_tmpl="{model}/{bpw}",
                w_tmpl="{model}/w-{bpw}",
                dry_run=False,
            )
            if not ok:
                print(f"🔴 Quantization failed for BPW {bpw}")
                return 1

            # Verify immediately: KL + PPL (parallel on 2+ GPUs)
            rc = run_measure_single_bpw(
                model_dir=model_dir,
                bpw=str(bpw),
                devices=measure_devices,
                db_path=db_path,
                ppl_rows=ppl_rows,
                write_logs=write_logs,
                include_base_ppl=(i == 0),
            )
            if rc != 0:
                print(f"🔴 Verification failed for BPW {bpw}")
                return 1

        # Stage 2: optimized optimize (needs all integer quants done)
        if optimized_bpws:
            _run_optimized_opt_stage(
                model_dir=model_dir,
                optimized_bpws=optimized_bpws,
                devices=devices,
                layers=optimized_measure_layers,
                write_logs=write_logs,
            )
            # Verify each optimized BPW
            for opt_bpw in optimized_bpws:
                rc = run_measure_single_bpw(
                    model_dir=model_dir,
                    bpw=str(opt_bpw),
                    devices=measure_devices,
                    db_path=db_path,
                    ppl_rows=ppl_rows,
                    write_logs=write_logs,
                )
                if rc != 0:
                    print(f"🔴 Verification failed for optimized BPW {opt_bpw}")
                    return 1

        # Export DB to CSV for downstream (readme, graph)
        export_csv(db_path, _out_csv)

        # Catbench runs after all verification is done
        if catbench_n > 0:
            rc = run_measure_stage(
                model_dir=model_dir,
                bpws=measure_bpws,
                devices=measure_devices,
                write_logs=write_logs,
                measure_args=measure_args,
                catbench_n=catbench_n,
            )
            if rc != 0:
                return rc

    else:
        # --- LEGACY MODE (--no-verify, or --no-measurement, or quant-only) ---

        # Stage 1: quantize all
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

        # Stage 2: optimized optimize
        if do_quant and optimized_bpws:
            _run_optimized_opt_stage(
                model_dir=model_dir,
                optimized_bpws=optimized_bpws,
                devices=devices,
                layers=optimized_measure_layers,
                write_logs=write_logs,
            )

        # Stage 3: measure all
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
        
        # 2. Legacy shard CSVs + measurement database
        gpu_csvs = glob.glob(os.path.join(model_dir, "*.gpu*.csv"))
        for f in gpu_csvs:
            print(f"  Removing legacy shard CSV {os.path.basename(f)}...")
            try: os.remove(f)
            except Exception as e: print(f"  🔴 Failed to remove {f}: {e}")
        db_files = glob.glob(os.path.join(model_dir, "*.db")) + glob.glob(os.path.join(model_dir, "*.db-wal")) + glob.glob(os.path.join(model_dir, "*.db-shm"))
        for f in db_files:
            print(f"  Removing {os.path.basename(f)}...")
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
