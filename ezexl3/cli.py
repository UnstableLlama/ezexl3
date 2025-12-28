#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

from ezexl3 import __version__
import json
from pathlib import Path

@dataclass
class PassThrough:
    quant_args: List[str]
    measure_args: List[str]
    cleaned_argv: List[str]

def save_exllamav3_root(path: str) -> None:
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
        p_sub.add_argument("-m", "--model", required=True, help="Path to BF16/base model directory")
        p_sub.add_argument(
            "--exllamav3-root",
            help="Path to exllamav3 checkout (saved for future runs)",
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
        p_sub.add_argument("--cleanup", action="store_true", help="Remove w-* working dirs after success")
        p_sub.add_argument("--no-quant", action="store_true", help="Skip quantization stage")
        p_sub.add_argument("--no-measure", action="store_true", help="Skip measurement stage")
        p_sub.add_argument("--no-report", action="store_true", help="Skip report stage")
        p_sub.add_argument("--no-meta", action="store_true", help="Do not write run.json receipt")
        p_sub.add_argument("--no-logs", action="store_true", help="Do not write per-GPU logs")

    # --- repo (main command) ---
    repo = sub.add_parser("repo", help="Generate an EXL3 repo (quantize -> measure -> report)")
    add_repo_flags(repo)

    # --- quantize ---
    q = sub.add_parser("quantize", help="Quantize only (vendored multiConvert)")
    q.add_argument("-m", "--models", nargs="+", required=True,
                   help="One or more input model directories (space or comma separated).")
    q.add_argument("-b", "--bpws", nargs="+", required=True,
                   help="Target BPWs (space or comma separated).")
    q.add_argument("--out-template", default="{model}/{bpw}",
                   help="Template for output directory. Fields: {model}, {model_name}, {bpw}")
    q.add_argument("--w-template", default="{model}/w-{bpw}",
                   help="Template for working directory. Fields: {model}, {model_name}, {bpw}")
    q.add_argument("--dry", action="store_true", help="Print what would run, but do not execute.")
    q.add_argument("--continue-on-error", action="store_true", help="Keep going after failures.")

    # --- measure ---
    m = sub.add_parser("measure", help="Measure only (vendored quantMeasure)")
    m.add_argument("-m", "--model", required=True, help="Path to BF16/base model directory")
    m.add_argument("-b", "--bpws", nargs="+", required=True, help="BPWs to measure (space or comma separated)")
    m.add_argument("-d", "--device", default="0", help="CUDA device index (default: 0)")
    m.add_argument("--csv", default=None, help="Override CSV output path")
    m.add_argument("--no-skip-done", action="store_true", help="Do not skip rows already in the CSV")
    m.add_argument(
        "--exllamav3-root",
        help="Path to exllamav3 checkout (saved for future runs)",
    )


    # Placeholders for later:
    sub.add_parser("report", help="Report only (CSV -> README/plots)")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # Extract passthrough blocks FIRST, then parse the cleaned argv normally.
    pt = _split_passthrough(argv)

    parser = build_parser()
    args = parser.parse_args(pt.cleaned_argv)

    # Normalize lists
    if getattr(args, "cmd", None) == "repo":
        args.bpws = _csv_or_space_list(args.bpws)
        # devices: "0,1" -> ["0","1"]
        args.devices = [d.strip() for d in str(args.devices).split(",") if d.strip()]
        if args.exllamav3_root:
            save_exllamav3_root(args.exllamav3_root)

        if args.device_ratios is not None:
            args.device_ratios = [x.strip() for x in str(args.device_ratios).split(",") if x.strip()]
        else:
            args.device_ratios = None

        # Attach passthrough
        args.quant_args = pt.quant_args
        args.measure_args = pt.measure_args

        from ezexl3.repo import run_repo

        # convert devices list[str] -> list[int]
        devices_i = [int(d) for d in args.devices]

        # device_ratios currently parsed into list[str] or None.
        # run_repo expects a single string like "1,1" (or None) to pass through to converter.
        device_ratios_str = ",".join(args.device_ratios) if args.device_ratios else None

        rc = run_repo(
            model_dir=args.model,
            bpws=args.bpws,
            devices=devices_i,
            device_ratios=device_ratios_str,
            quant_args=args.quant_args,
            measure_args=args.measure_args,
            exllamav3_root=args.exllamav3_root,   # ← THIS LINE
            do_quant=(not args.no_quant),
            do_measure=(not args.no_measure),
            do_report=(not args.no_report),
            cleanup=args.cleanup,
            write_logs=(not args.no_logs),
        )
        return rc


    if getattr(args, "cmd", None) == "quantize":
        from ezexl3.quantize import run as quant_run

        args.models = _csv_or_space_list(args.models)
        args.bpws = _csv_or_space_list(args.bpws)

        # Passthrough: use --quant-args -- ...
        forwarded = pt.quant_args

        rc = quant_run(
            models=args.models,
            bpws=args.bpws,
            forwarded=forwarded,
            out_template=args.out_template,
            w_template=args.w_template,
            dry_run=args.dry,
            continue_on_error=args.continue_on_error,
        )
        return rc
    if getattr(args, "cmd", None) == "measure":
        from ezexl3.measure import run_measure

        args.bpws = _csv_or_space_list(args.bpws)
        quants = args.bpws

        rc = run_measure(
            base_dir=args.model,
            quants=quants,
            device=int(args.device),
            csv_path=args.csv,
            skip_done=(not args.no_skip_done),
            exllamav3_root=args.exllamav3_root,   # ← SAME THING
        )

        return rc


    # Other subcommands not wired yet
    print(f"Command '{args.cmd}' not implemented yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
