# ezexl3/mtp.py
"""
mtp.py - Quantize just the MTP tensors from a base model checkpoint.

Wraps the vendored exllamav3 util/convert_mtp.py. Models quantized before
MTP support was added to exllamav3 don't contain the MTP tensors; the
output .safetensors file from this stage can be placed alongside a legacy
quant's other .safetensors files to enable MTP speculative decoding.
"""

from __future__ import annotations

import os
import sys

from ezexl3.measure import run_cmd_capture

_CONVERT_MTP_SCRIPT = os.path.join(os.path.dirname(__file__), "vendor", "convert_mtp.py")


def default_mtp_out_file(model_dir: str, mtp_bits: int, hq: bool = False) -> str:
    """Default output path: a subdirectory of the model dir, so the quantized
    MTP tensors never sit next to the base model's own .safetensors files
    (duplicate tensor keys would break loading the base model)."""
    suffix = "_hq" if hq else ""
    return os.path.join(
        os.path.abspath(model_dir), "mtp-quant", f"mtp_{mtp_bits}bpw{suffix}.safetensors"
    )


def run_mtp(
    model_dir: str,
    mtp_bits: int = 4,
    out_file: str | None = None,
    device: int = 0,
    hq: bool = False,
) -> int:
    """Quantize the MTP component of *model_dir* to *mtp_bits* and write the
    tensors to *out_file*. Skips if the output file already exists."""
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    if out_file is None:
        out_file = default_mtp_out_file(model_dir, mtp_bits, hq)
    out_file = os.path.abspath(out_file)

    if os.path.exists(out_file):
        print(f" -- MTP output already exists, skipping: {out_file}")
        print("    (delete the file to re-convert)")
        return 0

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    cmd = [
        sys.executable,
        _CONVERT_MTP_SCRIPT,
        "-m", model_dir,
        "-mb", str(mtp_bits),
        "-o", out_file,
        "-d", str(device),
    ]
    if hq:
        cmd.append("-hq")
    run_cmd_capture(cmd)

    print(f"\n -- MTP tensors written: {out_file}")
    print(" -- To add MTP support to a legacy quant, copy this file alongside")
    print("    the quant's other .safetensors files.")
    return 0
