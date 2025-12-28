#!/usr/bin/env python3
from __future__ import annotations

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
    # Placeholders for later:
    sub.add_parser("measure", help="Measure only (vendored quantMeasure)")
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
        if args.device_ratios is not None:
            args.device_ratios = [x.strip() for x in str(args.device_ratios).split(",") if x.strip()]
        else:
            args.device_ratios = None

        # Attach passthrough
        args.quant_args = pt.quant_args
        args.measure_args = pt.measure_args

        # For now, just echo the parsed plan (we'll wire real execution next)
        print("ezexl3 process (plan only for now)")
        print(f"  model        : {args.model}")
        print(f"  bpws         : {args.bpws}")
        print(f"  devices      : {args.devices}")
        print(f"  device_ratios: {args.device_ratios}")
        print(f"  schedule     : {args.schedule}")
        print(f"  quant_args   : {args.quant_args}")
        print(f"  measure_args : {args.measure_args}")
        print(f"  stages       : quant={not args.no_quant} measure={not args.no_measure} report={not args.no_report}")
        print(f"  cleanup      : {args.cleanup}")
        print(f"  meta/logs    : meta={not args.no_meta} logs={not args.no_logs}")
        print("")
        return 0

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

    # Other subcommands not wired yet
    print(f"Command '{args.cmd}' not implemented yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
