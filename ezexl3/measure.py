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
from typing import Dict, List, Set

CSV_FIELDS = ["weights", "KL Div", "PPL r-100", "GiB"]


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
    """Total .safetensors size in the immediate path in GiB (non-recursive)."""
    total = 0
    if not os.path.isdir(path):
        return 0.0
    for fn in os.listdir(path):
        if fn.endswith(".safetensors"):
            total += os.path.getsize(os.path.join(path, fn))
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
    full_out = "".join(out_lines)
    if rc != 0:
        raise RuntimeError(f"Command failed (rc={rc}): {' '.join(cmd)}\n\nOutput:\n{full_out}")
    return full_out

def find_model_diff_script() -> str:
    """
    Find internal model_diff.py script.
    """
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(pkg_dir, "model_diff.py")
    if os.path.exists(script_path):
        return script_path

    raise RuntimeError(f"Could not find local model_diff.py at {script_path}")


def run_model_diff(
    base_dir: str,
    other_dir: str,
    device: int,
    r: int = 10,
) -> float:
    """
    Runs internal model_diff.py and returns KL divergence.
    """
    script_path = find_model_diff_script()

    cmd = [
        sys.executable,
        script_path,
        "-ma", base_dir,
        "-mb", other_dir,
        "-r", str(r),
        "-d", str(device),
    ]
    out = run_cmd_capture(cmd)

    # Accept standard floats, scientific notation, and nan/inf.
    kl_match = re.search(
        r"(?:KL|K/L)\s+divergence(?:\s+\(A,\s+B\))?:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)",
        out,
        re.IGNORECASE,
    )

    if not kl_match:
        raise ValueError("Could not parse model_diff output (KL Divergence pattern did not match).")

    return float(kl_match.group(1))

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

    ppl_match = re.search(
        r"Perplexity:\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|inf|-inf)",
        out,
        re.IGNORECASE,
    )
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
    return_row: bool = False,
    ppl_rows: int = 100,
) -> int | dict:
    base_dir = os.path.abspath(base_dir)
    if csv_path is None:
        csv_path = default_csv_path(base_dir)
    csv_path = os.path.abspath(csv_path)

    ensure_csv_exists(csv_path)
    done = read_existing_weights(csv_path) if skip_done else set()

    for q in quants:
        if q == "base":
            model_dir = base_dir
            label = "bf16"
        else:
            model_dir = os.path.join(base_dir, str(q))
            label = str(q)

        if skip_done and label in done:
            # If return_row is True and we skip, we should probably still return the existing row?
            # For simplicity, we'll return None if skipped.
            if return_row and len(quants) == 1:
                return {} # Or find the row in the CSV
            continue

        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Quant dir not found: {model_dir}")

        if q == "base":
            kl_div = 0.0
            ppl_100 = run_ppl_layer(model_dir, device=device, r=ppl_rows)
            size_gib = file_size_gib(model_dir)
        else:
            kl_div = run_model_diff(
                base_dir,
                model_dir,
                device=device,
                r=10,
            )
            ppl_100 = run_ppl_layer(model_dir, device=device, r=ppl_rows)
            size_gib = file_size_gib(model_dir)

        row = {
            "weights": label,
            "KL Div": kl_div,
            "PPL r-100": ppl_100,
            "GiB": size_gib,
        }

        append_csv_row(csv_path, row)
        if return_row and len(quants) == 1:
            return row

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
