import argparse
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
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




def _parse_devices(values: List[str]) -> List[int]:
    if not values:
        raise SystemExit("At least one CUDA device must be provided (e.g. -d 0)")
    out: List[int] = []
    for raw in values:
        try:
            out.append(int(raw))
        except ValueError as e:
            raise SystemExit(f"Invalid CUDA device '{raw}'. Expected integer device ids like: -d 0,1") from e
    if not out:
        raise SystemExit("At least one CUDA device must be provided (e.g. -d 0)")
    return out


def _parse_device_ratios(values: Optional[List[str]], devices: List[int]) -> Optional[List[str]]:
    if values is None:
        return None
    if not values:
        raise SystemExit("--device-ratios cannot be empty. Example: -r 1,1")

    parsed: List[str] = []
    for raw in values:
        try:
            ratio = float(raw)
        except ValueError as e:
            raise SystemExit(f"Invalid device ratio '{raw}'. Expected numeric values like: -r 1,1") from e
        if ratio <= 0:
            raise SystemExit(f"Invalid device ratio '{raw}'. Ratios must be > 0")
        parsed.append(raw)

    if len(parsed) != len(devices):
        raise SystemExit(
            f"--device-ratios length ({len(parsed)}) must match --devices length ({len(devices)})."
        )

    return parsed



def _parse_layers(value: int) -> int:
    if value not in (1, 2, 3):
        raise SystemExit("--layers must be one of: 1, 2, 3")
    return value


def _parse_per_bpw_flag(
    flag_val: Optional[List[str]], all_bpws: List[str]
) -> Set[str]:
    """Parse a per-BPW flag (like -hq or -hb8) into a set of BPW strings.

    - None  → flag not provided → empty set
    - []    → bare flag (no args) → applies to ALL BPWs
    - ['4,6,8'] or ['4','6','8'] → specific BPWs
    """
    if flag_val is None:
        return set()
    if len(flag_val) == 0:
        # Bare flag: normalize all_bpws to the same string form
        return {_norm_bpw(b) for raw in all_bpws for b in raw.split(",")}
    # Explicit BPW list: flatten comma-separated values
    return {_norm_bpw(b) for raw in flag_val for b in raw.split(",")}


def _norm_bpw(b: str) -> str:
    """Normalize a BPW string for consistent comparison (strip whitespace)."""
    return b.strip()

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
        p_sub.add_argument("-hq", nargs="*", default=None,
                           help="Enable high-quality quantization (exllamav3 -hq). "
                                "Bare flag applies to all BPWs; with args applies to listed BPWs only. "
                                "Example: -hq 4,6,8")
        p_sub.add_argument("-hb8", nargs="*", default=None,
                           help="Use 8-bit head quantization (exllamav3 -hb 8) instead of default 6. "
                                "Bare flag applies to all BPWs; with args applies to listed BPWs only. "
                                "Example: -hb8 6,8")
        p_sub.add_argument("-opt", nargs="*", default=None,
                           help="Use optimized quantization pipeline for fractional BPWs. "
                                "Compares neighboring integer quants to find optimal mix. "
                                "Only applies to fractional BPWs. "
                                "Bare flag applies to all fractional BPWs; with args applies to listed only. "
                                "Example: -opt 4.5,5.5")
        p_sub.add_argument("-pm", action="store_true",
                           help="Parallel modules: speeds up quantization of MoE models. "
                                "Forwarded to exllamav3 multiConvert as -pm.")
        p_sub.add_argument("--no-cleanup", "-nc", action="store_true", help="Keep w-* working dirs and logs")
        p_sub.add_argument("--no-readme", action="store_true", help="Skip README stage")
        p_sub.add_argument("--no-logs", action="store_true", help="Do not write per-GPU logs")
        p_sub.add_argument("--no-prompt", "-np", action="store_true", help="Use defaults for README instead of prompting")
        p_sub.add_argument("--no-graph", "-ng", action="store_true", help="Do not generate or embed the README SVG graph")
        p_sub.add_argument("--no-measurement", "-nm", action="store_true", help="Skip KL/PPL measurements (also disables README graph and KL/PPL table columns)")
        p_sub.add_argument("--template", "-t", help="README template name (e.g., 'fire', 'basic')")
        p_sub.add_argument("-l", "--layers", type=int, default=2, choices=[1, 2, 3], help="Layers used by optimized comparative measure stage (1-3, default: 2)")
        p_sub.add_argument("--no-verify", "-nv", action="store_true",
                           help="Skip per-BPW verification (batch all quants, then batch all measures)")
        # Eval scripts (optional, a-la-carte)
        p_sub.add_argument("--no-kl", action="store_true",
                           help="Skip KL divergence measurement")
        p_sub.add_argument("--no-ppl", action="store_true",
                           help="Skip perplexity measurement")
        p_sub.add_argument("-cb", "--catbench", type=int, default=0, nargs="?", const=3,
                           help="Run SVG Catbench with N samples per model (default: 3 when flag present)")
        p_sub.add_argument("-div", "--diversity", type=int, default=0, nargs="?", const=50,
                           help="Run diversity eval with N samples (default: 50)")
        p_sub.add_argument("-he", "--humaneval", type=int, default=0, nargs="?", const=200,
                           help="Run HumanEval code gen eval with N samples/task (default: 200)")
        p_sub.add_argument("-ifb", "--ifbench", type=int, default=0, nargs="?", const=16384,
                           help="Run IFBench instruction following eval (default max_tokens: 16384)")
        p_sub.add_argument("-lctx", "--longctx", type=int, default=0, nargs="?", const=1,
                           help="Run long context understanding eval")
        p_sub.add_argument("-mmlu", "--mmlu", type=int, default=0, nargs="?", const=5,
                           help="Run MMLU knowledge benchmark with N fewshot examples (default: 5)")
        p_sub.add_argument("-perf", "--perf", type=int, default=0, nargs="?", const=32768,
                           help="Run inference performance benchmark (default max_length: 32768)")

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
    q.add_argument("-hq", nargs="*", default=None,
                   help="Enable high-quality quantization (exllamav3 -hq). "
                        "Bare flag applies to all BPWs; with args applies to listed BPWs only.")
    q.add_argument("-hb8", nargs="*", default=None,
                   help="Use 8-bit head quantization (exllamav3 -hb 8) instead of default 6. "
                        "Bare flag applies to all BPWs; with args applies to listed BPWs only.")
    q.add_argument("-opt", nargs="*", default=None,
                   help="Use optimized quantization pipeline for fractional BPWs. "
                        "Only applies to fractional BPWs. "
                        "Bare flag applies to all fractional BPWs; with args applies to listed only.")
    q.add_argument("-pm", action="store_true",
                   help="Parallel modules: speeds up quantization of MoE models. "
                        "Forwarded to exllamav3 multiConvert as -pm.")
    q.add_argument("--dry", action="store_true", help="Print what would run, but do not execute.")
    q.add_argument("--continue-on-error", action="store_true", help="Keep going after failures.")
    q.add_argument("--no-logs", action="store_true", help="Do not write per-GPU logs")
    q.add_argument("-l", "--layers", type=int, default=2, choices=[1, 2, 3], help="Layers used by optimized comparative measure stage (1-3, default: 2)")

    # --- measure ---
    m = sub.add_parser("measure", help="Measure only (vendored quantMeasure)")
    m.add_argument("-m", "--models", nargs="+", required=True, help="One or more model directories")
    m.add_argument("-b", "--bpws", nargs="+", required=True, help="BPWs to measure (space or comma separated)")
    m.add_argument("-d", "--devices", default="0", help="CUDA devices for measurement. Example: -d 0,1")
    m.add_argument("--no-logs", action="store_true", help="Do not write per-GPU logs")
    m.add_argument("--no-cleanup", "-nc", action="store_true", help="Keep temporary shard CSVs and logs")
    # Eval scripts (optional, a-la-carte)
    m.add_argument("--no-kl", action="store_true",
                   help="Skip KL divergence measurement")
    m.add_argument("--no-ppl", action="store_true",
                   help="Skip perplexity measurement")
    m.add_argument("-cb", "--catbench", type=int, default=0, nargs="?", const=3,
                   help="Run SVG Catbench with N samples per model (default: 3 when flag present)")
    m.add_argument("-div", "--diversity", type=int, default=0, nargs="?", const=50,
                   help="Run diversity eval with N samples (default: 50)")
    m.add_argument("-he", "--humaneval", type=int, default=0, nargs="?", const=200,
                   help="Run HumanEval code gen eval with N samples/task (default: 200)")
    m.add_argument("-ifb", "--ifbench", type=int, default=0, nargs="?", const=16384,
                   help="Run IFBench instruction following eval (default max_tokens: 16384)")
    m.add_argument("-lctx", "--longctx", type=int, default=0, nargs="?", const=1,
                   help="Run long context understanding eval")
    m.add_argument("-mmlu", "--mmlu", type=int, default=0, nargs="?", const=5,
                   help="Run MMLU knowledge benchmark with N fewshot examples (default: 5)")
    m.add_argument("-perf", "--perf", type=int, default=0, nargs="?", const=32768,
                   help="Run inference performance benchmark (default max_length: 32768)")


    # --- chat ---
    ch = sub.add_parser("chat", help="Launch web chat UI (model optional — select in browser)")
    ch.add_argument("-m", "--model", required=False, default=None, help="Model directory (optional: select in UI)")
    ch.add_argument("-d", "--devices", default="0", help="CUDA devices. Example: -d 0,1")
    ch.add_argument("-r", "--device-ratios", default=None, help="Device ratios. Example: -r 1,1")
    ch.add_argument("--host", default="127.0.0.1",
                    help="Bind address (default: 127.0.0.1). WARNING: non-loopback addresses expose the unauthenticated API to the network")
    ch.add_argument("--port", type=int, default=8800, help="Port (default: 8800)")
    ch.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    ch.add_argument("-cs", "--cache-size", type=int, default=None,
                    help="Cache size in tokens (default: 32768). Must be multiple of 256")
    ch.add_argument("-cq", "--cache-quant", type=str, default=None,
                    help="Cache quantization bits: kv_bits or k_bits,v_bits (default: 6,6)")

    # --- readme ---
    r = sub.add_parser("readme", help="README only (CSV -> README)")
    r.add_argument("-m", "--models", nargs="+", required=True, help="One or more model directories")
    r.add_argument("-b", "--bpws", nargs="+", default=None, help="BPWs (required for single mode, auto-detected otherwise)")
    r.add_argument("--mode", choices=["branched", "single"], default="single",
                   help="single: per-BPW READMEs with cross-linked repos. branched: single README")
    r.add_argument("--no-prompt", "-np", action="store_true", help="Use defaults for README instead of prompting")
    r.add_argument("--no-graph", "-ng", action="store_true", help="Do not generate or embed the README SVG graph")
    r.add_argument("--no-measurement", "-nm", action="store_true", help="Remove KL/PPL columns from README and skip graph embedding")
    r.add_argument("--template", "-t", help="README template name (e.g., 'fire', 'basic')")

    # --- upload ---
    u = sub.add_parser("upload", help="Upload quantized models to HuggingFace")
    u.add_argument("-m", "--models", nargs="+", required=True, help="One or more model directories")
    u.add_argument("-b", "--bpws", nargs="+", required=True, help="BPWs to upload (space or comma separated)")
    u.add_argument("--mode", choices=["branched", "single"], default="single",
                   help="single: separate repo per BPW. branched: single repo with branches per BPW")
    u.add_argument("--private", action="store_true", help="Create private HuggingFace repos")
    u.add_argument("--small-only", action="store_true",
                   help="Exclude large files (*.safetensors, *.bin, *.pt, *.ckpt)")
    u.add_argument("--create-only", action="store_true",
                   help="Only create repos/branches, do not upload files")
    u.add_argument("-dr", "--dry-run", action="store_true",
                   help="Preview the repos that would be created without contacting HuggingFace")

    # --- ui (dashboard) ---
    ui = sub.add_parser("ui", aliases=["dash", "dashboard"],
                        help="Launch dashboard web UI")
    ui.add_argument("--host", default="127.0.0.1",
                    help="Bind address (default: 127.0.0.1)")
    ui.add_argument("--port", type=int, default=8801, help="Port (default: 8801)")
    ui.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")

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

    if cmd == "chat":
        from ezexl3.chat.server import run_server
        # When -m is omitted, launch UI-only (no model pre-loaded)
        chat_devices = None
        dr = None
        if args.model:
            chat_devices = [int(d) for d in args.devices]
            dr = args.device_ratios
            if dr:
                dr = ",".join(dr) if isinstance(dr, list) else dr
        run_server(
            model_dir=args.model,
            devices=chat_devices,
            device_ratios=dr,
            cache_size=args.cache_size,
            cache_quant=args.cache_quant,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
        return 0

    if cmd in ("ui", "dash", "dashboard"):
        from ezexl3.ui.server import run_ui_server
        run_ui_server(
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
        )
        return 0

    from ezexl3.repo import run_repo, run_quant_stage, run_measure_stage

    devices_i = _parse_devices(getattr(args, "devices", ["0"]))
    device_ratios = _parse_device_ratios(getattr(args, "device_ratios", None), devices_i)
    device_ratios_str = ",".join(device_ratios) if device_ratios else None
    layers = _parse_layers(getattr(args, "layers", 2)) if hasattr(args, "layers") else 2

    # -pm (parallel modules / MoE speedup) is forwarded as a real flag
    # to multiConvert via the existing quant_args passthrough pipeline.
    if getattr(args, "pm", False) and "-pm" not in pt.quant_args:
        pt.quant_args = list(pt.quant_args) + ["-pm"]

    # Build per-BPW flag sets from -hq, -hb8, and -opt
    hq_bpws = _parse_per_bpw_flag(getattr(args, "hq", None), args.bpws)
    hb8_bpws = _parse_per_bpw_flag(getattr(args, "hb8", None), args.bpws)
    opt_bpws = _parse_per_bpw_flag(getattr(args, "opt", None), args.bpws)
    # -opt only applies to fractional BPWs; silently drop any integers
    opt_bpws = {b for b in opt_bpws if "." in b}

    # When a fractional BPW is painted with -opt, the actual quantization
    # happens on its integer neighbors (e.g. 4.5 → quantize 4 and 5, then
    # combine). Propagate any -hq / -hb8 paints from the fractional onto
    # those donor integers so the donors are built with the requested
    # quality flag.
    import math as _math
    def _neighbors(frac: str) -> List[str]:
        try:
            v = float(frac)
        except ValueError:
            return []
        return [str(int(_math.floor(v))), str(int(_math.ceil(v)))]
    for frac in opt_bpws:
        nbrs = _neighbors(frac)
        if frac in hq_bpws:
            hq_bpws.update(nbrs)
        if frac in hb8_bpws:
            hb8_bpws.update(nbrs)

    # Collect enabled eval flags into a dict: {name: arg_value}
    _EVAL_FLAG_NAMES = ["diversity", "humaneval", "ifbench", "longctx", "mmlu", "perf"]
    enabled_evals = {}
    for name in _EVAL_FLAG_NAMES:
        val = getattr(args, name, 0) or 0
        if val:
            enabled_evals[name] = val

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
                    do_measure=(not args.no_measurement),
                    do_readme=(not args.no_readme),
                    cleanup=(not args.no_cleanup),
                    write_logs=(not args.no_logs),
                    interactive=(not args.no_prompt),
                    include_graph=(not args.no_graph and not args.no_measurement),
                    include_measurements=(not args.no_measurement),
                    template=args.template,
                    optimized_measure_layers=layers,
                    catbench_n=getattr(args, "catbench", 0) or 0,
                    verify=(not args.no_verify),
                    evals=enabled_evals or None,
                    skip_kl=getattr(args, "no_kl", False),
                    skip_ppl=getattr(args, "no_ppl", False),
                    hq_bpws=hq_bpws,
                    hb8_bpws=hb8_bpws,
                    opt_bpws=opt_bpws,
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
        from ezexl3.repo import _plan_repo_bpws, _run_optimized_opt_stage

        bpw_plan = _plan_repo_bpws(args.bpws, opt_bpws=opt_bpws)
        quant_bpws = bpw_plan["quant_integer_queue"]
        optimized_bpws = bpw_plan["requested_optimizeds"]

        if optimized_bpws and args.out_template != "{model}/{bpw}":
            print("Error: --out-template cannot be customized when using decimal BPWs.")
            print("The optimized quantization stage requires outputs at {model}/{bpw}.")
            return 1

        all_requested = set(bpw_plan["requested_integers"] + bpw_plan.get("requested_optimizeds", []))
        # Also exclude standard fractional BPWs (in quant queue but explicitly requested)
        all_requested.update(b for raw in args.bpws for b in raw.split(",") if b.strip())
        auto_added = [b for b in quant_bpws if b not in all_requested]
        if auto_added:
            print(
                "ℹ️ Added required integer quants for -opt targets: "
                + ", ".join(auto_added)
            )

        failed_models: List[str] = []
        for model_dir in args.models:
            print(f"\nQuantizing model: {model_dir}")
            try:
                rc = run_quant_stage(
                    model_dir=model_dir,
                    bpws=quant_bpws,
                    devices=devices_i,
                    device_ratios=device_ratios_str,
                    quant_args=pt.quant_args,
                    out_template=args.out_template,
                    w_template=args.w_template,
                    dry_run=args.dry,
                    continue_on_error=args.continue_on_error,
                    optimized_measure_layers=layers,
                    hq_bpws=hq_bpws,
                    hb8_bpws=hb8_bpws,
                )
                if rc != 0:
                    failed_models.append(model_dir)
                    continue

                if optimized_bpws and not args.dry:
                    _run_optimized_opt_stage(
                        model_dir=os.path.abspath(model_dir),
                        optimized_bpws=optimized_bpws,
                        devices=devices_i,
                        layers=layers,
                        write_logs=not args.no_logs,
                    )
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
                    measure_args=pt.measure_args,
                    catbench_n=getattr(args, "catbench", 0) or 0,
                    evals=enabled_evals or None,
                    skip_kl=getattr(args, "no_kl", False),
                    skip_ppl=getattr(args, "no_ppl", False),
                )
                if rc != 0:
                    failed_models.append(model_dir)
            except Exception as e:
                print(f"Error measuring {model_dir}: {e}")
                failed_models.append(model_dir)
        return 1 if failed_models else 0

    if cmd == "readme":
        from ezexl3.readme import run_readme, run_readme_single
        readme_bpws = args.bpws if hasattr(args, "bpws") and args.bpws else None
        readme_mode = getattr(args, "mode", "branched")
        for model_dir in args.models:
            if readme_mode == "single":
                run_readme_single(
                    model_dir,
                    bpws=readme_bpws,
                    template_name=args.template,
                    interactive=(not args.no_prompt),
                    include_graph=(not args.no_graph and not args.no_measurement),
                    include_measurements=(not args.no_measurement),
                )
            else:
                run_readme(
                    model_dir,
                    template_name=args.template,
                    interactive=(not args.no_prompt),
                    include_graph=(not args.no_graph and not args.no_measurement),
                    include_measurements=(not args.no_measurement),
                    bpws_hint=readme_bpws,
                )
        return 0

    if cmd == "upload":
        from ezexl3.upload import run_upload
        failed_models: List[str] = []
        for model_dir in args.models:
            try:
                rc = run_upload(
                    model_dir=model_dir,
                    bpws=args.bpws,
                    mode=args.mode,
                    private=args.private,
                    small_only=args.small_only,
                    create_only=args.create_only,
                    dry_run=args.dry_run,
                )
                if rc != 0:
                    failed_models.append(model_dir)
            except Exception as e:
                print(f"Error uploading {model_dir}: {e}")
                import traceback
                traceback.print_exc()
                failed_models.append(model_dir)
        return 1 if failed_models else 0

    print(f"Command '{args.cmd}' not implemented yet.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
