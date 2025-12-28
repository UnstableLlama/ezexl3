# ezexl3/measure.py
#!/usr/bin/env python3
"""
measure.py - Automated quantization quality measurement for EXL3 / exllamav3

This is a minimal port of your quantMeasure.py into the ezexl3 package.

Additions vs original:
- Optional --csv path override (for per-GPU shard CSVs)
- Generic skip-if-done (skip rows already present in the target CSV)
- Callable entrypoint: run_measure(...)
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from typing import Dict, List, Set, Tuple
import importlib.util
import json
from pathlib import Path

EZEXL3_CONFIG_DIR = Path.home() / ".config" / "ezexl3"
EZEXL3_CONFIG_FILE = EZEXL3_CONFIG_DIR / "config.json"


def load_ezexl3_config() -> dict:
    if not EZEXL3_CONFIG_FILE.exists():
        return {}
    try:
        with open(EZEXL3_CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def save_ezexl3_config(cfg: dict) -> None:
    EZEXL3_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(EZEXL3_CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def resolve_exllamav3_root(cli_value: str | None = None) -> str | None:
    """
    Resolution order:
      1) CLI flag
      2) saved config
      3) EXLLAMAV3_ROOT env var
      4) None
    """
    if cli_value:
        cfg = load_ezexl3_config()
        cfg["exllamav3_root"] = cli_value
        save_ezexl3_config(cfg)
        return cli_value

    cfg = load_ezexl3_config()
    if "exllamav3_root" in cfg:
        return cfg["exllamav3_root"]

    env = os.environ.get("EXLLAMAV3_ROOT")
    if env:
        return env

    return None

CSV_FIELDS = ["weights", "PPL r-10", "K/L Div", "PPL r-100", "GiB"]


# ----------------------------
# Helpers: filesystem + CSV
# ----------------------------

def base_dir_name(base_dir: str) -> str:
    return os.path.basename(os.path.abspath(base_dir.rstrip("/")))


def default_csv_path(base_dir: str) -> str:
    return os.path.join(base_dir, f"{base_dir_name(base_dir)}Measured.csv")


def ensure_csv_exists(csv_path: str) -> None:
    """Create CSV + header immediately if it doesn't exist."""
    if os.path.exists(csv_path):
        return
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())


def read_existing_weights(csv_path: str) -> Set[str]:
    """Return the set of 'weights' values already present in csv_path."""
    if not os.path.exists(csv_path):
        return set()
    out: Set[str] = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return set()
        for row in reader:
            w = (row.get("weights") or "").strip()
            if w:
                out.add(w)
    return out


def append_csv_row(csv_path: str, row: Dict[str, object]) -> None:
    """Append a row to the CSV and fsync immediately."""
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def file_size_gib(path: str) -> float:
    """Total .safetensors size under path in GiB."""
    total = 0
    for root, _, files in os.walk(path):
        for fn in files:
            if fn.endswith(".safetensors"):
                total += os.path.getsize(os.path.join(root, fn))
    return total / (1024 ** 3)


# ----------------------------
# Helpers: CLI parsing
# ----------------------------

def parse_quants_str(s: str) -> List[str]:
    # Accept comma or space separated
    parts: List[str] = []
    for token in s.replace(",", " ").split():
        t = token.strip()
        if t:
            parts.append(t)
    return parts


# ----------------------------
# Helpers: running eval tools
# ----------------------------

def run_cmd_capture(cmd: List[str]) -> str:
    """Run a command, stream stdout live, and also capture full output."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    out_lines: List[str] = []
    for line in proc.stdout:
        sys.stdout.write(line)
        out_lines.append(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}")
    return "".join(out_lines)

def find_model_diff_script(exllamav3_root: str | None = None) -> str:
    exllamav3_root = resolve_exllamav3_root(exllamav3_root)
    """
    Find exllamav3 eval/model_diff.py across common layouts.

    Search order:
      1) explicit exllamav3_root/eval/model_diff.py (if provided)
      2) $EXLLAMAV3_ROOT/eval/model_diff.py
      3) dev checkout layout: (pkg_dir/..)/eval/model_diff.py
      4) packaged layout: pkg_dir/eval/model_diff.py
    """
    script_name = "model_diff.py"

    # 1) Explicit root
    if exllamav3_root:
        cand = os.path.join(os.path.abspath(exllamav3_root), "eval", script_name)
        if os.path.exists(cand):
            return cand

    # 2) Env var root
    env_root = os.environ.get("EXLLAMAV3_ROOT", "").strip()
    if env_root:
        cand = os.path.join(os.path.abspath(env_root), "eval", script_name)
        if os.path.exists(cand):
            return cand

    # 3/4) Auto-detect from installed package location
    spec = importlib.util.find_spec("exllamav3")
    if spec and spec.submodule_search_locations:
        pkg_dir = list(spec.submodule_search_locations)[0]  # .../exllamav3 OR .../site-packages/exllamav3

        # Dev checkout usually has: repo_root/exllamav3 (pkg) and repo_root/eval (scripts)
        repo_root = os.path.abspath(os.path.join(pkg_dir, os.pardir))
        cand_repo = os.path.join(repo_root, "eval", script_name)
        if os.path.exists(cand_repo):
            return cand_repo

        # Some installs might put eval scripts inside the package dir
        cand_pkg = os.path.join(pkg_dir, "eval", script_name)
        if os.path.exists(cand_pkg):
            return cand_pkg

    raise RuntimeError(
        "Could not find exllamav3 eval script 'model_diff.py'.\n"
        "Fix: set EXLLAMAV3_ROOT to your exllamav3 checkout directory.\n"
        "Example:\n"
        "  export EXLLAMAV3_ROOT=/home/you/path/to/exllamav3\n"
    )


def run_model_diff(
    base_dir: str,
    other_dir: str,
    device: int,
    r: int = 10,
    exllamav3_root: str | None = None,
) -> Tuple[float, float]:
    """
    Runs exllamav3's model_diff.py (wherever it lives) and returns:
      (ppl_r10, kl_div)
    """
    script_path = find_model_diff_script(exllamav3_root)

    cmd = [
        sys.executable,
        script_path,
        "-ma", base_dir,
        "-mb", other_dir,
        "-r", str(r),
        "-d", str(device),
    ]
    out = run_cmd_capture(cmd)

    ppl_match = re.search(r"Perplexity:\s+([\d.]+)", out)
    kl_match = re.search(r"K/L Divergence:\s+([\d.]+)", out)
    if not ppl_match:
        raise ValueError("Could not parse model_diff output (Perplexity pattern didn't match).")
    if not kl_match:
        raise ValueError("Could not parse model_diff output (K/L Divergence pattern didn't match).")

    return float(ppl_match.group(1)), float(kl_match.group(1))

def run_ppl_layer(model_dir: str, device: int, r: int = 100) -> float:
    """
    Runs bundled ezexl3.ppl_layer and returns PPL.
    """
    cmd = [
        sys.executable,
        "-m",
        "ezexl3.ppl_layer",
        "-m", model_dir,
        "-r", str(r),
        "-d", str(device),
    ]
    out = run_cmd_capture(cmd)

    ppl_match = re.search(r"Perplexity:\s+([\d.]+)", out)
    if not ppl_match:
        raise ValueError("Could not parse ppl_layer output (Perplexity pattern didn't match).")

    return float(ppl_match.group(1))



# ----------------------------
# Main callable: run_measure
# ----------------------------

def run_measure(
    base_dir: str,
    quants: List[str],
    device: int = 0,
    csv_path: str | None = None,
    skip_done: bool = True,
    exllamav3_root: str | None = None,
) -> int:
    base_dir = os.path.abspath(base_dir)
    if csv_path is None:
        csv_path = default_csv_path(base_dir)
    csv_path = os.path.abspath(csv_path)

    ensure_csv_exists(csv_path)
    done = read_existing_weights(csv_path) if skip_done else set()

    print(f"\n{'='*60}")
    print("Quant Measurement")
    print(f"{'='*60}")
    print(f"Base   : {base_dir}")
    print(f"Device : {device}")
    print(f"CSV    : {csv_path}")
    print(f"Items  : {quants}")
    print("=" * 60)

    for q in quants:
        if q == "base":
            model_dir = base_dir
            label = "bf16"
        else:
            model_dir = os.path.join(base_dir, str(q))
            label = str(q)

        if skip_done and label in done:
            print(f"🟦 skipping: already measured ({label})")
            continue

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Quant dir not found: {model_dir}")

        if q == "base":
            print(f"\n--- Measuring base (bf16) ---")
            ppl_r10 = ""
            kl_div = 0.0
            ppl_100 = run_ppl_layer(model_dir, device=device, r=100)
            size_gib = file_size_gib(model_dir)
        else:
            print(f"\n--- Measuring quant {q} ---")
            ppl_r10, kl_div = run_model_diff(
                base_dir,
                model_dir,
                device=device,
                r=10,
                exllamav3_root=exllamav3_root,
            )
            ppl_100 = run_ppl_layer(model_dir, device=device, r=100)
            size_gib = file_size_gib(model_dir)

        row = {
            "weights": label,
            "PPL r-10": ppl_r10,
            "K/L Div": kl_div,
            "PPL r-100": ppl_100,
            "GiB": size_gib,
        }

        append_csv_row(csv_path, row)
        print(f"✓ Wrote row for {label}")

    print(f"\n{'='*60}")
    print("Measurement Complete")
    print(f"{'='*60}")
    print(f"Results saved to: {csv_path}")
    print("=" * 60)

    return 0

# ----------------------------
# CLI entry
# ----------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Measure quantization quality across multiple quants")
    parser.add_argument("-b", "--base", type=str, required=True, help="Base model directory (BF16 directory)")
    parser.add_argument(
        "-q",
        "--quants",
        type=str,
        required=True,
        help='Items to measure (comma or space separated). Include "base" to add BF16 row. Example: -q "base 2 3 4"',
    )
    parser.add_argument("-d", "--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--csv", type=str, default=None, help="Override output CSV path (for per-GPU shards)")
    parser.add_argument("--no-skip-done", action="store_true", help="Do not skip rows already in the CSV")
    args = parser.parse_args()

    quants = parse_quants_str(args.quants)
    return run_measure(
        base_dir=args.base,
        quants=quants,
        device=args.device,
        csv_path=args.csv,
        skip_done=(not args.no_skip_done),
    )


if __name__ == "__main__":
    raise SystemExit(main())
