import argparse
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional
from ezexl3 import __version__

@dataclass
class PassThrough:
    quant_args: List[str]
    measure_args: List[str]
    cleaned_argv: List[str]

def save_exllamav3_root(path: str) -> None:
    import json
    from pathlib import Path
    cfg_dir = Path.home() / ".config" / "ezexl3"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = cfg_dir / "config.json"
    with cfg_path.open("w") as f:
        json.dump({"exllamav3_root": path}, f, indent=2)

def _split_passthrough(argv: List[str]) -> PassThrough:
    """
    Extract two passthrough blocks:

      --quant-args -- <...>
      --measure-args -- <...>

    Everything else stays in cleaned_argv for normal argparse parsing.

    Notes:
    - The '--' delimiter is REQUIRED if you supply a block.
    - Order can be either (quant then measure) or (measure then quant), but
      each block may appear at most once.
    """
    quant_args: List[str] = []
    measure_args: List[str] = []

    cleaned: List[str] = []
    i = 0
    n = len(argv)

    def read_block(start_i: int) -> Tuple[List[str], int]:
        # Expect: <flag> -- <args...>
        if start_i + 1 >= n or argv[start_i + 1] != "--":
            raise SystemExit(
                f"Expected '--' after {argv[start_i]}. Example: {argv[start_i]} -- -d 0,1 -dr 1,1"
            )
        j = start_i + 2
        block: List[str] = []
        while j < n and argv[j] not in ("--quant-args", "--measure-args"):
            block.append(argv[j])
            j += 1
        return block, j

    while i < n:
        tok = argv[i]
        if tok == "--quant-args":
            if quant_args:
                raise SystemExit("Duplicate --quant-args block")
            block, i = read_block(i)
            quant_args = block
            continue
        if tok == "--measure-args":
            if measure_args:
                raise SystemExit("Duplicate --measure-args block")
            block, i = read_block(i)
            measure_args = block
            continue

        cleaned.append(tok)
        i += 1

    return PassThrough(quant_args=quant_args, measure_args=measure_args, cleaned_argv=cleaned)


def _csv_or_space_list(values: List[str]) -> List[str]:
    """
    Allows: -b 2 3 4 5 6
    or:     -b 2,3,4,5,6
    """
    out: List[str] = []
    for v in values:
        parts = [p.strip() for p in v.split(",") if p.strip()]
        out.extend(parts)
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ezexl3",
        description="ezexl3: simple single-command EXL3 repo generator"
    )
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_repo_flags(p_sub: argparse.ArgumentParser) -> None:
        p_sub.add_argument("-m", "--models", nargs="+", required=True,
                           help="One or more BF16/base model directories (space or comma separated)")
        p_sub.add_argument(
            "--exllamav3-root",
            help="[DEPRECATED] No longer needed, bundled model_diff.py is used.",
        )
        p_sub.add_argument(
            "-b", "--bpws",
            required=True,
            nargs="+",
            help="Target BPWs (space-separated or comma-separated). Example: -b 2 3 4 5 6 or -b 2,3,4,5,6",
        )
        p_sub.add_argument(
            "-d", "--devices",
            default="0",
            help="CUDA devices for quant+measure. Example: -d 0,1",
        )
        p_sub.add_argument(
            "-r", "--device-ratios",
            default=None,
            help="Device ratios for quantization only. Example: -r 1,1 (optional)",
        )
        p_sub.add_argument("--schedule", choices=["queue", "static"], default="queue",
                           help="Measurement scheduling strategy (default: queue)")
        p_sub.add_argument("--no-cleanup", "-nc", action="store_true", help="Keep w-* working dirs and logs")
        p_sub.add_argument("--no-readme", action="store_true", help="Skip README stage")
        p_sub.add_argument("--no-meta", action="store_true", help="Do not write run.json receipt")
        p_sub.add_argument("--no-logs", action="store_true", help="Do not write per-GPU logs")
        p_sub.add_argument("--no-prompt", "-np", action="store_true", help="Use defaults for README instead of prompting")
        p_sub.add_argument("--template", "-t", help="README template name (e.g., 'fire', 'basic')")

    # --- repo (main command) ---
    repo = sub.add_parser("repo", help="Generate an EXL3 repo (quantize -> measure -> README)")
    add_repo_flags(repo)

    # --- quantize ---
    q = sub.add_parser("quantize", aliases=["quant"], help="Quantize only (vendored multiConvert)")
    q.add_argument("-m", "--models", nargs="+", required=True,
                   help="One or more input model directories (space or comma separated).")
    q.add_argument("-b", "--bpws", nargs="+", required=True,
                   help="Target BPWs (space or comma separated).")
    q.add_argument("-d", "--devices", default="0", help="CUDA devices. Example: -d 0,1")
    q.add_argument("-r", "--device-ratios", default=None, help="Device ratios. Example: -r 1,1")
    q.add_argument("--out-template", default="{model}/{bpw}",
                   help="Template for output directory. Fields: {model}, {model_name}, {bpw}")
    q.add_argument("--w-template", default="{model}/w-{bpw}",
                   help="Template for working directory. Fields: {model}, {model_name}, {bpw}")
    q.add_argument("--dry", action="store_true", help="Print what would run, but do not execute.")
    q.add_argument("--continue-on-error", action="store_true", help="Keep going after failures.")

    # --- measure ---
    m = sub.add_parser("measure", help="Measure only (vendored quantMeasure)")
    m.add_argument("-m", "--models", nargs="+", required=True, help="One or more model directories")
    m.add_argument("-b", "--bpws", nargs="+", required=True, help="BPWs to measure (space or comma separated)")
    m.add_argument("-d", "--devices", default="0", help="CUDA devices for measurement. Example: -d 0,1")
    m.add_argument("--no-logs", action="store_true", help="Do not write per-GPU logs")
    m.add_argument("--no-cleanup", "-nc", action="store_true", help="Keep temporary shard CSVs and logs")
    m.add_argument(
        "--exllamav3-root",
        help="[DEPRECATED] No longer needed, bundled model_diff.py is used.",
    )


    # --- readme ---
    r = sub.add_parser("readme", help="README only (CSV -> README)")
    r.add_argument("-m", "--models", nargs="+", required=True, help="One or more model directories")
    r.add_argument("--no-prompt", "-np", action="store_true", help="Use defaults for README instead of prompting")
    r.add_argument("--template", "-t", help="README template name (e.g., 'fire', 'basic')")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Extract passthrough blocks FIRST, then parse the cleaned argv normally.
    pt = _split_passthrough(argv)

    parser = build_parser()
    args = parser.parse_args(pt.cleaned_argv)

    # Initialize common fields
    cmd = getattr(args, "cmd", None)
    if not cmd:
        parser.print_help()
        return 0

    # Normalize lists
    if hasattr(args, "models"):
        args.models = _csv_or_space_list(args.models)
    if hasattr(args, "bpws"):
        args.bpws = _csv_or_space_list(args.bpws)
    if hasattr(args, "devices"):
        args.devices = [d.strip() for d in str(args.devices).split(",") if d.strip()]
    if hasattr(args, "device_ratios") and args.device_ratios is not None:
        args.device_ratios = [x.strip() for x in str(args.device_ratios).split(",") if x.strip()]
    

    import os
    from ezexl3.repo import run_repo, run_quant_stage, run_measure_stage

    devices_i = [int(d) for d in getattr(args, "devices", ["0"])]
    device_ratios_str = ",".join(args.device_ratios) if getattr(args, "device_ratios", None) else None

    if cmd == "repo":
        # Process each model, continuing on error
        failed_models: List[str] = []
        for model_dir in args.models:
            model_name = os.path.basename(os.path.abspath(model_dir))
            print(f"\n{'='*60}")
            print(f"Processing model: {model_name}")
            print(f"{'='*60}")

            try:
                rc = run_repo(
                    model_dir=model_dir,
                    bpws=args.bpws,
                    devices=devices_i,
                    device_ratios=device_ratios_str,
                    quant_args=pt.quant_args,
                    measure_args=pt.measure_args,
                    do_quant=True,
                    do_measure=True,
                    do_readme=(not args.no_readme),
                    cleanup=(not args.no_cleanup),
                    write_logs=(not args.no_logs),
                    interactive=(not args.no_prompt),
                    template=args.template,
                )
                if rc != 0:
                    failed_models.append(model_dir)
            except Exception as e:
                print(f"Error processing {model_name}: {e}")
                import traceback
                traceback.print_exc()
                failed_models.append(model_dir)

        if failed_models:
            print(f"\n{'='*60}")
            print(f"Completed with {len(failed_models)} failure(s): {failed_models}")
            print(f"{'='*60}")
            return 1
        return 0

    if cmd in ("quant", "quantize"):
        failed_models: List[str] = []
        for model_dir in args.models:
            print(f"\nQuantizing model: {model_dir}")
            try:
                rc = run_quant_stage(
                    model_dir=model_dir,
                    bpws=args.bpws,
                    devices=devices_i,
                    device_ratios=device_ratios_str,
                    quant_args=pt.quant_args,
                )
                if rc != 0:
                    failed_models.append(model_dir)
            except Exception as e:
                print(f"Error quantizing {model_dir}: {e}")
                failed_models.append(model_dir)
        return 1 if failed_models else 0

    if cmd == "measure":
        failed_models: List[str] = []
        for model_dir in args.models:
            print(f"\nMeasuring model: {model_dir}")
            try:
                rc = run_measure_stage(
                    model_dir=model_dir,
                    bpws=args.bpws,
                    devices=devices_i,
                    write_logs=(not args.no_logs),
                )
                if rc != 0:
                    failed_models.append(model_dir)
            except Exception as e:
                print(f"Error measuring {model_dir}: {e}")
                failed_models.append(model_dir)
        return 1 if failed_models else 0

    if cmd == "readme":
        from ezexl3.readme import run_readme
        for model_dir in args.models:
            run_readme(model_dir, template_name=args.template, interactive=(not args.no_prompt))
        return 0

    print(f"Command '{args.cmd}' not implemented yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
