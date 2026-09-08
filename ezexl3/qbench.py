# ezexl3/qbench.py
"""
qbench.py - Run exllamav3's qbench quantization-comparison harness.

Wraps the vendored eval/qbench.py (upstream's replacement for compare_q.py).
A YAML project file is generated from the ezexl3 model-directory layout
(base model + <bpw>/ quant subdirs), then the harness runs it: the
reference model's logits are computed once and cached to disk, every quant
is streamed through the same test data against the cache, and a second
reference pass with BF16-rounding noise gives the model's self-noise floor.
KLD is reported as mean/median/p90 plus confidence buckets, with scatter,
spread and histogram plots.

The default test data is a separate self-generated evaluation trace, sampled
once with qbench_prompts.py and then reused. SC calibration traces are not
automatically selected. Explicit --dataset/--trace overrides remain available.

The generated project file lives at <model>/qbench/project.yml and is
REUSED on later runs unless --regen is given, so it can be hand-edited —
e.g. to add GGUF entries (llamacpp engine), HF checkpoints (transformers
engine), or extra options. Because results are cached per (test data,
reference, model), re-running after adding one entry only measures that
entry.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from typing import Dict, List, Optional

from ezexl3.measure import run_cmd_capture

_QBENCH_SCRIPT = os.path.join(os.path.dirname(__file__), "vendor", "eval", "qbench.py")
_TRACE_SCRIPT = os.path.join(os.path.dirname(__file__), "vendor", "eval", "qbench_prompts.py")

# Output files written next to project.yml (relative paths in the project
# resolve against the project file's directory)
_OUTPUT_FILES = {
    "results": "qb_results.json",
    "plot_ppl": "qb_ppl.png",
    "plot_kld": "qb_kld.png",
    "plot_kld_spread": "qb_kld_spread.png",
    "plot_kld_hist": "qb_kld_hist.png",
    "plot_kld_hist_combined": "qb_kld_hist_combined.png",
}

# The charts the generated README embeds, in display order: mean KLD vs bpw,
# perplexity vs bpw, then the per-token KLD panels. Copied up to the model
# root by publish_charts() so uploads never have to reach into qbench/, which
# also holds the (very large) logit cache.
README_CHARTS = ["qb_kld.png", "qb_ppl.png", "qb_kld_hist.png", "qb_kld_hist_combined.png"]

_TRACE_CANDIDATES = (
    "qbench/qbench_prompts_gen.json", "qbench_prompts_gen.json",
)
_DATA_FIELDS = ("test_trace", "test_data", "tokenizer")


def _trace_summary(path: str) -> str:
    """Validate qbench's JSON format; packed SC safetensors lack response boundaries."""
    with open(path, encoding="utf8") as f:
        trace = json.load(f)
    if not isinstance(trace, dict) or type(trace.get("vocab_size")) is not int or trace["vocab_size"] <= 0:
        raise ValueError("trace needs a positive vocab_size")
    rows = trace.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("trace needs nonempty rows")
    inputs = outputs = max_length = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each trace row must be an object")
        for field in ("input_ids", "response_ids"):
            ids = row.get(field)
            if not isinstance(ids, list) or not ids or any(type(t) is not int or t < 0 for t in ids):
                raise ValueError(f"each trace row needs nonempty {field} token IDs")
        inputs += len(row["input_ids"])
        outputs += len(row["response_ids"])
        max_length = max(max_length, len(row["input_ids"]) + len(row["response_ids"]))
    return f"{len(rows):,} rows, {inputs:,} input + {outputs:,} response tokens; longest row {max_length:,}"


def ensure_eval_trace(model_dir: str, device: int, trace_devices: Optional[List[int]] = None) -> str:
    """Reuse a dedicated eval trace, or generate one with upstream's eval prompts."""
    for rel in _TRACE_CANDIDATES:
        path = os.path.join(model_dir, rel)
        if not os.path.isfile(path):
            continue
        try:
            _trace_summary(path)
        except (OSError, ValueError) as e:
            raise ValueError(f"Invalid evaluation trace {path}: {e}") from e
        print(f" -- Reusing dedicated evaluation trace: {path}")
        return path

    from ezexl3.repo_selfcal import _TRACE_DONOR_MIN_BPW, _find_trace_donor, _run_script

    donor = _find_trace_donor(model_dir) or model_dir
    devices = list(trace_devices) if trace_devices is not None else [device]
    if not devices:
        raise ValueError("Evaluation trace generation requires at least one GPU")
    visible_devices = ",".join(str(d) for d in devices)
    path = os.path.join(model_dir, _TRACE_CANDIDATES[0])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f" -- Generating a separate evaluation trace with qbench_prompts.py from: {donor}")
    print(f" -- Trace generation GPUs: {visible_devices}; model and KV cache automatically split across these GPUs.")
    print(" -- Upstream defaults: target 20,000 response tokens, up to 4,096 per turn, 30% tool conversations.")
    if donor == model_dir:
        print(f" -- No completed >= {_TRACE_DONOR_MIN_BPW:g} bpw donor found; using the base model (full generation load across the selected GPUs).")
    tmp = path + ".tmp"
    _run_script(
        [sys.executable, _TRACE_SCRIPT, "-m", donor, "-o", tmp,
         "--min_tokens", "20000", "--max_new_tokens", "4096", "--tool_frac", "0.3"],
        env_extra={"CUDA_VISIBLE_DEVICES": visible_devices},
        log_path=os.path.join(os.path.dirname(path), "eval_trace.log"),
    )
    _trace_summary(tmp)
    os.replace(tmp, path)
    print(f" -- Saved evaluation trace: {path}; later runs reuse these same samples.")
    return path


def _data_fields(project: dict) -> dict:
    return {k: project[k] for k in _DATA_FIELDS if k in project}


def _configure_test_source(project: dict, model_dir: str, project_path: str,
                           trace: Optional[str], dataset: Optional[str],
                           rows: int, length: int, template: str, fresh: bool, device: int,
                           trace_devices: Optional[List[int]] = None) -> None:
    """Auto-upgrade generated defaults, while preserving hand-edited test settings."""
    current = _data_fields(project)
    legacy_default = _data_fields(build_project(model_dir, []))
    generated = False
    if not fresh and os.path.isfile(project_path):
        with open(project_path, encoding="utf8") as f:
            generated = f.readline().startswith("# qbench project generated by ezexl3.")
    automatic = fresh or project.get("_ezexl3_auto_data") == current or (
        generated and "_ezexl3_auto_data" not in project
        and not project.get("_ezexl3_explicit_data") and current == legacy_default
    )
    if trace is not None or dataset is not None or automatic:
        selected = os.path.abspath(trace) if trace else (
            ensure_eval_trace(model_dir, device, trace_devices) if dataset is None else None
        )
        if selected:
            _trace_summary(selected)  # Explicit invalid traces fail; never silently substitute.
        replacement = build_project(model_dir, [], rows=rows, length=length,
                                    dataset=dataset or "wiki2", template=template, trace=selected)
        for key in _DATA_FIELDS:
            project.pop(key, None)
        project.update(_data_fields(replacement))
        if trace is None and dataset is None:
            project["_ezexl3_auto_data"] = json.loads(json.dumps(_data_fields(project)))
            project.pop("_ezexl3_explicit_data", None)
        else:
            project.pop("_ezexl3_auto_data", None)
            project["_ezexl3_explicit_data"] = True
        if current != _data_fields(project) and not fresh:
            print(" -- Test source changed; measurements for this test source use a separate cache.")
    else:
        print(" -- Keeping the existing project's test settings.")

    if project.get("test_trace"):
        path = project["test_trace"]
        if not os.path.isabs(path):
            path = os.path.normpath(os.path.join(os.path.dirname(project_path), path))
        print(f" -- QBench test source: self-generated in-domain trace: {path}")
        print(f" -- {_trace_summary(path)}; scoring response positions (dataset/rows/length do not apply).")
        if os.path.basename(path) == "cal_trace.json":
            print(" -- Reusing SC calibration data for evaluation; this is not an independent eval trace.")
    else:
        td = project["test_data"]
        print(f" -- QBench test source: dataset {td['source']}, {td['rows']} rows x {td['length']} tokens.")

# Labels build_project() gives EXL3 quant entries, e.g. "4 bpw" / "4.5 bpw".
_BPW_LABEL_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*bpw$", re.IGNORECASE)


def default_qbench_dir(model_dir: str) -> str:
    return os.path.join(os.path.abspath(model_dir), "qbench")


def default_project_path(model_dir: str) -> str:
    return os.path.join(default_qbench_dir(model_dir), "project.yml")


def check_qbench_support() -> Optional[str]:
    """Return an error string when qbench can't run in this environment."""
    missing = []
    for mod in ("yaml", "seaborn"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        pkgs = " ".join("pyyaml" if m == "yaml" else m for m in missing)
        return (
            f"qbench needs {', '.join(missing)} which "
            f"{'is' if len(missing) == 1 else 'are'} not installed. "
            f"Fix with: pip install {pkgs}"
        )
    try:
        from transformers import TokenizersBackend  # noqa: F401
    except ImportError:
        return (
            "qbench needs transformers>=5.0.0 for TokenizersBackend chat tokenizers. "
            "Upgrade in the environment running ezexl3: "
            f'{sys.executable} -m pip install -U "transformers>=5.0.0"'
        )
    try:
        from exllamav3.util import measures  # noqa: F401
    except ImportError:
        return (
            "The installed exllamav3 does not provide exllamav3.util.measures — "
            "qbench needs a recent exllamav3 (>= 1.4)."
        )
    return None


def build_project(
    model_dir: str,
    bpws: List[str],
    rows: int = 10,
    length: int = 2048,
    dataset: str = "wiki2",
    template: str = "none",
    trace: Optional[str] = None,
    ref_engine: str = "exllamav3",
    cache_gb: float = 50.0,
    noise_floor: bool = True,
) -> dict:
    """Build a qbench project dict for an ezexl3 model directory."""
    model_dir = os.path.abspath(model_dir)
    project: dict = {"title": os.path.basename(model_dir)}

    if trace:
        project["test_trace"] = os.path.abspath(trace)
    else:
        project["test_data"] = {
            "source": "openwebtext10k" if dataset == "openwebtext" else dataset,
            "rows": rows,
            "length": length,
            "stride": length,
        }
        project["tokenizer"] = {
            "source": model_dir,
            # qbench accepts true / false / "assistant"
            "template": {"none": False, "chat": True, "assistant": "assistant"}[template],
        }

    project["logit_cache"] = {
        "dir": "logit_cache",
        "max_size_gb": cache_gb,
    }

    reference: dict = {
        "label": "BF16",
        "group": "reference",
        "engine": ref_engine,
        "source": model_dir,
    }
    if ref_engine == "transformers":
        reference["options"] = {"streaming": True}

    models = [reference]
    for bpw in bpws:
        models.append({
            "label": f"{bpw} bpw",
            "group": "EXL3",
            "engine": "exllamav3",
            "source": os.path.join(model_dir, bpw),
        })
    project["models"] = models

    project["noise_floor"] = noise_floor
    project["output"] = dict(_OUTPUT_FILES)
    return project


def write_project(path: str, project: dict) -> None:
    import yaml

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        f.write(
            "# qbench project generated by ezexl3.\n"
            "# Reused on later runs (hand-edits survive; --regen overwrites).\n"
            "# Add GGUF entries with engine: llamacpp, HF checkpoints with\n"
            "# engine: transformers. Relative paths resolve against this file.\n"
        )
        yaml.safe_dump(project, f, sort_keys=False, default_flow_style=False)


def _missing_quants(project: dict, model_dir: str, bpws: List[str]) -> List[str]:
    """BPWs whose quant dir is not referenced by any model entry in *project*."""
    sources = {
        os.path.normpath(str(m.get("source", "")))
        for m in project.get("models", [])
    }
    model_dir = os.path.abspath(model_dir)
    return [
        b for b in bpws
        if os.path.normpath(os.path.join(model_dir, b)) not in sources
    ]


def sync_project_models(project: dict, model_dir: str, bpws: List[str]) -> List[str]:
    """Add model entries for any of *bpws* the project doesn't reference yet.

    Existing entries are left alone, so hand edits (GGUF entries, extra HF
    checkpoints, per-model options) survive. This is what lets the repo
    pipeline measure incrementally: each newly quantized BPW is appended and
    qbench re-measures only that one, since results are cached per model.

    Returns the BPWs that were added.
    """
    missing = _missing_quants(project, model_dir, bpws)
    if not missing:
        return []
    model_dir = os.path.abspath(model_dir)
    models = project.setdefault("models", [])
    for bpw in missing:
        models.append({
            "label": f"{bpw} bpw",
            "group": "EXL3",
            "engine": "exllamav3",
            "source": os.path.join(model_dir, bpw),
        })
    return missing


def read_results(model_dir: str) -> Dict[str, dict]:
    """Parse qb_results.json into {csv label -> result dict}.

    Keys are measure-DB labels ("bf16", "4", "4.5"). The noise-floor row and
    any hand-added entries whose labels we don't generate are skipped — they
    have no BPW directory to attach to.
    """
    path = os.path.join(default_qbench_dir(model_dir), _OUTPUT_FILES["results"])
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf8") as f:
        results = json.load(f)

    out: Dict[str, dict] = {}
    for res in results:
        group = res.get("group")
        if group == "reference":
            out["bf16"] = res
            continue
        if group != "EXL3":
            continue  # noise_floor, GGUF/HF entries added by hand
        m = _BPW_LABEL_RE.match(str(res.get("label", "")))
        if m:
            out[m.group(1)] = res
    return out


def publish_charts(model_dir: str) -> List[str]:
    """Copy the README's charts from qbench/ up to the model root.

    The README references them as plain filenames, and upload only collects
    root-level artifacts — so the logit cache under qbench/ is never uploaded.
    Returns the filenames actually copied.
    """
    qb_dir = default_qbench_dir(model_dir)
    copied = []
    for name in README_CHARTS:
        src = os.path.join(qb_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(os.path.abspath(model_dir), name))
            copied.append(name)
    return copied


def run_qbench(
    model_dir: str,
    bpws: Optional[List[str]] = None,
    device: int = 0,
    rows: int = 10,
    length: int = 2048,
    dataset: Optional[str] = None,
    template: str = "none",
    trace: Optional[str] = None,
    ref_engine: str = "exllamav3",
    cache_gb: float = 50.0,
    noise_floor: bool = True,
    regen: bool = False,
    trace_devices: Optional[List[int]] = None,
) -> int:
    """Generate (or reuse) the project file for *model_dir* and run qbench."""
    model_dir = os.path.abspath(model_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    err = check_qbench_support()
    if err:
        print(f"🔴 {err}")
        return 1

    from ezexl3.readme import _discover_bpws

    if not bpws:
        bpws = _discover_bpws(model_dir)
        if not bpws:
            print("🔴 No BPWs specified and no quant subdirectories found "
                  f"in {model_dir}")
            return 1
        print(f" -- Auto-detected quants: {', '.join(bpws)}")

    missing = [b for b in bpws if not os.path.isdir(os.path.join(model_dir, b))]
    if missing:
        print(f"🔴 No quant directory for BPW(s): {', '.join(missing)}")
        return 1

    project_path = default_project_path(model_dir)
    if os.path.isfile(project_path) and not regen:
        import yaml
        with open(project_path, "r", encoding="utf8") as f:
            project = yaml.safe_load(f)
        before = json.dumps(project, sort_keys=True)
        print(f" -- Reusing existing project: {project_path}")
        _configure_test_source(project, model_dir, project_path, trace, dataset,
                               rows, length, template, fresh=False, device=device, trace_devices=trace_devices)
        project.setdefault("output", {}).setdefault("plot_kld_hist_combined", "qb_kld_hist_combined.png")
        added = sync_project_models(project, model_dir, bpws)
        if before != json.dumps(project, sort_keys=True):
            write_project(project_path, project)
        if added:
            print(f" -- Added quant(s) to the project: {', '.join(added)}")
    else:
        project = build_project(
            model_dir, bpws,
            rows=rows, length=length, dataset=dataset or "wiki2", template=template,
            trace=trace, ref_engine=ref_engine, cache_gb=cache_gb,
            noise_floor=noise_floor,
        )
        _configure_test_source(project, model_dir, project_path, trace, dataset,
                               rows, length, template, fresh=True, device=device, trace_devices=trace_devices)
        write_project(project_path, project)
        print(f" -- Wrote project: {project_path}")

    cmd = [sys.executable, _QBENCH_SCRIPT, project_path, "-d", str(device)]
    run_cmd_capture(cmd)

    qb_dir = default_qbench_dir(model_dir)
    print(f"\n -- qbench outputs in: {qb_dir}")
    copied = publish_charts(model_dir)
    if copied:
        print(f" -- Charts copied to the model root: {', '.join(copied)}")
    return 0
