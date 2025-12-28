# ezexl3/quantize.py
from __future__ import annotations

import os
import time
from typing import List, Tuple, Optional

from exllamav3.conversion.convert_model import parser as convert_parser
from exllamav3.conversion.convert_model import main as convert_main
from exllamav3.conversion.convert_model import prepare as convert_prepare


def _split_commas(items: List[str]) -> List[str]:
    out: List[str] = []
    for it in items:
        parts = [p.strip() for p in it.split(",") if p.strip()]
        out.extend(parts)
    return out


def _format_path(tmpl: str, model_dir: str, bpw: str) -> str:
    model_dir = model_dir.rstrip("/")
    model_name = os.path.basename(model_dir)
    return tmpl.format(model=model_dir, model_name=model_name, bpw=bpw)


def run_one(
    model_dir: str,
    bpw: str,
    forwarded: List[str],
    out_tmpl: str,
    w_tmpl: str,
    dry_run: bool,
) -> bool:
    out_dir = _format_path(out_tmpl, model_dir, bpw)
    w_dir = _format_path(w_tmpl, model_dir, bpw)

    # 1) Always skip if completed output exists
    if os.path.isdir(out_dir) and os.path.isfile(os.path.join(out_dir, "config.json")):
        print("🟦 skipping: output already exists")
        return True

    # 2) Auto-resume if workdir looks like a real job (args.json exists)
    resume_marker = os.path.join(w_dir, "args.json")
    if os.path.isdir(w_dir) and os.path.isfile(resume_marker):
        job_argv = ["-w", w_dir, "-r"] + forwarded
        print("\n============================================================")
        print("🔁 RESUMING JOB")
        print(f"Work  : {w_dir}")
        print(f"Args  : {' '.join(job_argv)}")
        print("============================================================")
    else:
        # 3) New job
        job_argv = ["-i", model_dir, "-o", out_dir, "-w", w_dir, "-b", str(bpw)] + forwarded
        print("\n============================================================")
        print("🚀 STARTING JOB")
        print(f"Model : {model_dir}")
        print(f"BPW   : {bpw}")
        print(f"Out   : {out_dir}")
        print(f"Work  : {w_dir}")
        print(f"Args  : {' '.join(job_argv)}")
        print("============================================================")

    if dry_run:
        print("🟡 dry-run: not executing")
        return True

    # Parse using the real exllamav3 convert parser, then call prepare/main like convert.py does.
    args = convert_parser.parse_args(job_argv)
    in_args, job_state, ok, err = convert_prepare(args)
    if not ok:
        print(f"🔴 prepare() failed: {err}")
        return False

    convert_main(in_args, job_state)
    print("🟢 done")
    return True


def run(
    models: List[str],
    bpws: List[str],
    forwarded: Optional[List[str]] = None,
    out_template: str = "{model}/{bpw}",
    w_template: str = "{model}/w-{bpw}",
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> int:
    """
    Run sequential EXL3 quantization jobs for each (model, bpw).

    Returns process-like exit code:
      0 = success (or continued through failures with continue_on_error)
      1 = stopped early on first failure
    """
    forwarded = forwarded or []

    models = _split_commas(models)
    bpws = _split_commas([str(b) for b in bpws])

    jobs: List[Tuple[str, str]] = [(m, b) for m in models for b in bpws]
    failures: List[Tuple[str, str]] = []

    start = time.time()
    total = len(jobs)

    for idx, (m, b) in enumerate(jobs, 1):
        print(f"\n🚀 Job {idx}/{total}")
        ok = run_one(m, b, forwarded, out_template, w_template, dry_run=dry_run)
        if not ok:
            failures.append((m, b))
            print(f"🔴 FAILED: {m} @ {b}")
            if not continue_on_error:
                elapsed = time.time() - start
                print(f"\nStopped early after {elapsed:.1f}s. Failures: {len(failures)}")
                return 1

    elapsed = time.time() - start
    print("\n============================================================")
    print("✅ ALL JOBS COMPLETE")
    print(f"Elapsed: {elapsed/60:.1f} min")
    if failures:
        print(f"Failures ({len(failures)}):")
        for m, b in failures:
            print(f"  - {m} @ {b}")
    print("============================================================")
    return 0
