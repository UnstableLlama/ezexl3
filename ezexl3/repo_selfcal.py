# ezexl3/repo_selfcal.py
#
# Self-calibrated quantization stage, wrapping exllamav3's experimental
# optimization pipeline (doc/optimize.md upstream):
#
#   sc_trace.py      -> self-sampled, in-domain calibration trace
#   sc_rfn_probe.py  -> per-tensor quantization error anchors (optional)
#   sc_measure.py    -> per-tensor sensitivity via shaped noise injection
#   sc_optimize.py   -> per-tensor bitrate recipe (YAML) per target BPW
#   convert          -> -rcp recipe.yaml -cd cal_trace.safetensors
#
# Every stage writes plain files under <model>/selfcal/ and is skipped when
# its output already exists, so interrupted runs resume where they left off.
from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, List, Optional, Tuple

_VENDOR_DIR = os.path.join(os.path.dirname(__file__), "vendor")
_SC_TRACE_SCRIPT = os.path.join(_VENDOR_DIR, "sc_trace.py")
_SC_RFN_PROBE_SCRIPT = os.path.join(_VENDOR_DIR, "sc_rfn_probe.py")
_SC_MEASURE_SCRIPT = os.path.join(_VENDOR_DIR, "sc_measure.py")
_SC_OPTIMIZE_SCRIPT = os.path.join(_VENDOR_DIR, "sc_optimize.py")

# The trace should come from a minimally-distorted model; upstream recommends
# the unquantized model or a >= 5-6 bpw quant (much faster than bf16).
_TRACE_DONOR_MIN_BPW = 5.0


def check_selfcal_support() -> Optional[str]:
    """Return an error message when the installed exllamav3 cannot convert
    with a recipe (-rcp) and external calibration data (-cd); None when OK."""
    try:
        from exllamav3.conversion.convert_model import parser as convert_parser
    except Exception as e:  # pragma: no cover - import environment specific
        return f"exllamav3 is not importable: {e}"
    opts = {s for a in convert_parser._actions for s in a.option_strings}
    if "-rcp" not in opts or "-cd" not in opts:
        return (
            "The installed exllamav3 does not support recipe-based conversion "
            "(-rcp/-cd). Self-calibrated quants (-sc) need exllamav3 >= 1.4.3 "
            "or the dev branch."
        )
    return None


def _selfcal_paths(model_dir: str) -> dict:
    sc_dir = os.path.join(model_dir, "selfcal")
    return {
        "dir": sc_dir,
        "trace_json": os.path.join(sc_dir, "cal_trace.json"),
        "trace_st": os.path.join(sc_dir, "cal_trace.safetensors"),
        "rfn": os.path.join(sc_dir, "rfn_probe.json"),
        "attrib": os.path.join(sc_dir, "noise_attrib.json"),
        "attrib_done": os.path.join(sc_dir, "noise_attrib.done"),
    }


def _recipe_path(model_dir: str, bpw: str) -> str:
    return os.path.join(model_dir, "selfcal", f"recipe_{bpw}bpw.yaml")


def _existing_quants(model_dir: str) -> List[Tuple[float, str, str]]:
    """(bpw_value, bpw_name, path) for every completed quant at <model>/<bpw>."""
    out: List[Tuple[float, str, str]] = []
    if not os.path.isdir(model_dir):
        return out
    for name in sorted(os.listdir(model_dir)):
        full = os.path.join(model_dir, name)
        if not os.path.isdir(full) or not os.path.isfile(os.path.join(full, "config.json")):
            continue
        try:
            value = float(name)
        except ValueError:
            continue
        if 0 < value <= 8:
            out.append((value, name, full))
    return out


def _find_trace_donor(model_dir: str, min_bpw: float = _TRACE_DONOR_MIN_BPW) -> Optional[str]:
    """Highest-bitrate completed quant >= min_bpw, or None (caller then uses
    the unquantized model, which is equivalent but slower)."""
    candidates = [(v, path) for v, _n, path in _existing_quants(model_dir) if v >= min_bpw]
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])[1]


def _find_probe_anchor(model_dir: str) -> Optional[Tuple[str, str]]:
    """Lowest-bitrate completed integer quant, used to anchor the noise model.
    Upstream validated K=2 anchors as extrapolating across the full K range."""
    candidates = [
        (v, name, path)
        for v, name, path in _existing_quants(model_dir)
        if float(v).is_integer()
    ]
    if not candidates:
        return None
    v, name, path = min(candidates, key=lambda t: t[0])
    return name, path


def _run_script(
    cmd: List[str],
    env_extra: Optional[dict] = None,
    log_path: Optional[str] = None,
) -> None:
    """Run a vendored pipeline script, echoing output and optionally teeing to
    a log file. Raises RuntimeError on a non-zero exit."""
    env = os.environ.copy()
    env["PYTHONSAFEPATH"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if env_extra:
        env.update(env_extra)

    log_f = None
    if log_path:
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        log_f = open(log_path, "a")
        log_f.write(f"$ {' '.join(cmd)}\n")
        log_f.flush()
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )
        assert proc.stdout is not None
        while True:
            # read1(), not read(): read() blocks until it has the full 4096
            # bytes, which stalls low-volume stages (sc_trace emits ~100 bytes
            # a minute) behind a buffer that takes hours to fill.
            chunk = proc.stdout.read1(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            sys.stdout.write(text)
            sys.stdout.flush()
            if log_f:
                log_f.write(text)
                log_f.flush()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed with exit code {rc}: {' '.join(cmd)}")
    finally:
        if log_f:
            log_f.close()


def run_selfcal_stage(
    model_dir: str,
    sc_bpws: List[str],
    devices: List[int],
    forwarded_for_bpw: Callable[[str], List[str]],
    head_bits: Optional[int] = None,
    write_logs: bool = True,
    trace_donor: Optional[str] = None,
    run_script_fn: Callable = _run_script,
    quant_one_fn: Optional[Callable] = None,
    check_support_fn: Callable = check_selfcal_support,
    executable: str = sys.executable,
) -> None:
    """Build every -sc painted BPW through the self-calibration pipeline.

    forwarded_for_bpw(bpw) must return the standard forwarded quant args for
    that BPW (devices, ratios, -hq/-hb paints); -rcp/-cd are appended here.

    trace_donor overrides the model the calibration trace is sampled from;
    None auto-picks per _find_trace_donor and falls back to the bf16 model.
    """
    if not sc_bpws:
        return
    if not devices:
        raise ValueError("No CUDA devices available for the self-calibration stage")

    err = check_support_fn()
    if err:
        raise RuntimeError(err)

    if quant_one_fn is None:
        from ezexl3.repo_subprocess import _run_quant_one_isolated
        quant_one_fn = _run_quant_one_isolated

    model_dir = os.path.abspath(model_dir)
    paths = _selfcal_paths(model_dir)
    os.makedirs(paths["dir"], exist_ok=True)
    logs_dir = os.path.join(model_dir, "logs")

    def _log(name: str) -> Optional[str]:
        return os.path.join(logs_dir, name) if write_logs else None

    # Pin the pipeline scripts to the selected GPUs; the scripts themselves
    # then address them as devices 0..N-1.
    device_env = {"CUDA_VISIBLE_DEVICES": ",".join(str(d) for d in devices)}

    print("\n============================================================")
    print(f"🧠 Self-calibrated quantization: {', '.join(sc_bpws)} bpw")
    print("============================================================")

    # --- Stage 1: self-sampled trace ---
    if os.path.isfile(paths["trace_st"]) and os.path.isfile(paths["trace_json"]):
        print(f"🟦 skipping trace: {os.path.basename(paths['trace_st'])} already exists")
    else:
        if trace_donor:
            donor = os.path.abspath(trace_donor)
            if not os.path.isfile(os.path.join(donor, "config.json")):
                raise RuntimeError(
                    f"Trace donor {donor} is not a model directory (no config.json)"
                )
            print(f"📝 Generating self-sampled trace from {donor} (explicit donor)")
        else:
            donor = _find_trace_donor(model_dir)
            if donor:
                print(f"📝 Generating self-sampled trace from {os.path.basename(donor)} bpw quant")
            else:
                donor = model_dir
                print("📝 Generating self-sampled trace from the unquantized model "
                      f"(no >= {_TRACE_DONOR_MIN_BPW:g} bpw quant found; this is slower)")
        run_script_fn(
            [
                executable, _SC_TRACE_SCRIPT,
                "-m", donor,
                "-o", paths["trace_json"],
                "-co", paths["trace_st"],
            ],
            env_extra=device_env,
            log_path=_log("selfcal_trace.log"),
        )

    # --- Stage 2: rfn probe (optional but recommended) ---
    rfn_path: Optional[str] = paths["rfn"] if os.path.isfile(paths["rfn"]) else None
    if rfn_path:
        print(f"🟦 skipping rfn probe: {os.path.basename(paths['rfn'])} already exists")
    else:
        anchor = _find_probe_anchor(model_dir)
        if anchor is None:
            print("⚠️  No existing integer quant to anchor the noise model — "
                  "sc_optimize will fall back to its global anchor")
        else:
            anchor_name, anchor_dir = anchor
            print(f"🔬 Probing per-tensor quantization error of the {anchor_name} bpw quant")
            run_script_fn(
                [
                    executable, _SC_RFN_PROBE_SCRIPT,
                    "-mq", anchor_dir,
                    "-mr", model_dir,
                    "-d", "0",
                    "-o", paths["rfn"],
                ],
                env_extra=device_env,
                log_path=_log("selfcal_rfn_probe.log"),
            )
            rfn_path = paths["rfn"]

    # --- Stage 3: sensitivity measurement (resumes from a partial file) ---
    if os.path.isfile(paths["attrib_done"]):
        print(f"🟦 skipping sensitivity measure: {os.path.basename(paths['attrib'])} already complete")
    else:
        print("📐 Measuring per-tensor sensitivity (shaped noise injection)")
        cmd = [
            executable, _SC_MEASURE_SCRIPT,
            "-m", model_dir,
            "-d", "0",
            "--shaped",
            "-tr", paths["trace_st"],
            "-rs", "1.0,0.5",
            "-o", paths["attrib"],
        ]
        if rfn_path:
            cmd += ["-rr", rfn_path]
        run_script_fn(cmd, env_extra=device_env, log_path=_log("selfcal_measure.log"))
        with open(paths["attrib_done"], "w") as f:
            f.write("ok\n")

    # --- Stage 4+5: per-BPW recipe + conversion ---
    for bpw in sc_bpws:
        out_dir = os.path.join(model_dir, bpw)
        if os.path.isdir(out_dir) and os.path.isfile(os.path.join(out_dir, "config.json")):
            print(f"🟦 skipping self-calibrated {bpw} bpw: output already exists")
            continue

        recipe = _recipe_path(model_dir, bpw)
        if os.path.isfile(recipe):
            print(f"🟦 skipping recipe {bpw} bpw: {os.path.basename(recipe)} already exists")
        else:
            print(f"⚙️  Compiling bitrate recipe for {bpw} bpw")
            cmd = [
                executable, _SC_OPTIMIZE_SCRIPT,
                "-m", paths["attrib"],
                "-b", bpw,
                "-al", "2.0",
                "-o", recipe,
            ]
            if rfn_path:
                cmd += ["-rr", rfn_path]
            if head_bits is not None:
                cmd += ["-hb", str(head_bits)]
            run_script_fn(cmd, log_path=_log("selfcal_optimize.log"))

        print(f"🚀 Converting self-calibrated {bpw} bpw quant")
        forwarded = list(forwarded_for_bpw(bpw)) + [
            "-rcp", recipe,
            "-cd", paths["trace_st"],
        ]
        ok = quant_one_fn(
            model_dir, bpw, forwarded,
            out_tmpl="{model}/{bpw}",
            w_tmpl="{model}/w-{bpw}",
        )
        if not ok:
            raise RuntimeError(f"Self-calibrated conversion failed for {bpw} bpw")
