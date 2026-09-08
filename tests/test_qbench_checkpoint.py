"""Exercise the vendored cache and runner with tiny CPU tensors and fake inference."""
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1] / "ezexl3/vendor/eval"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def harness(monkeypatch, tmp_path):
    misc = types.ModuleType("exllamav3.util.misc")
    misc.prepend_hf_chat_context = Mock()
    monkeypatch.setitem(sys.modules, misc.__name__, misc)
    data = load_module("qbench.data", ROOT / "qbench/data.py")
    monkeypatch.setitem(sys.modules, "qbench.data", data)
    engines = types.ModuleType("qbench.engines")
    calls = []
    fail = set()

    def open_backend(spec, *args):
        label = spec["label"]
        calls.append(label)
        if label in fail:
            raise RuntimeError("interrupted")
        backend = Mock(info={"bpw": 4})
        backend.run.side_effect = lambda ids, callback, **kw: callback(0, torch.zeros(1))
        return backend

    engines.open_backend = open_backend
    monkeypatch.setitem(sys.modules, engines.__name__, engines)
    measure = types.ModuleType("qbench.measure")
    measure.BF16_ROUNDING_EPS = 2 ** -9
    measure.METRICS_VERSION = 5
    measure.print_stats = Mock()
    measure.DiffStats = lambda *args: Mock(**{
        "results.return_value": {"ppl": 2, "kld": 0.1},
        "kl_vector.return_value": torch.zeros(1),
    })

    def save_row(store, row, logits, ranges, conf):
        data.save_tensors(str(Path(store) / f"row_{row:06d}.safetensors"), {"logits": logits})
        conf.append(torch.zeros(1))

    measure.save_reference_row = save_row
    monkeypatch.setitem(sys.modules, measure.__name__, measure)
    plot = types.ModuleType("qbench.plot")
    for name in ("plot_kld_hist", "plot_kld_hist_combined", "plot_kld_spread", "plot_scatter"):
        setattr(plot, name, Mock())
    monkeypatch.setitem(sys.modules, plot.__name__, plot)
    # The script appends import paths for standalone execution.
    monkeypatch.setattr(sys, "path", list(sys.path))
    runner = load_module("checkpoint_runner", ROOT / "qbench.py")
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"vocab_size": 2, "rows": [{"input_ids": [0], "response_ids": [1]}]}))
    project = {
        "test_trace": str(trace), "noise_floor": True,
        "logit_cache": {"dir": str(tmp_path / "cache")},
        "models": [{"source": str(model), "label": "BF16", "engine": "exllamav3", "group": "reference"}],
        "output": {"results": str(tmp_path / "results.json")},
    }

    def add_quant(label):
        path = model / label
        path.mkdir()
        (path / "model.safetensors").write_bytes(b"weights")
        project["models"].append({"source": str(path), "label": label, "engine": "exllamav3", "group": "EXL3"})
        return path

    def run():
        calls.clear()
        path = tmp_path / "project.yml"
        path.write_text(yaml.safe_dump(project))
        runner.main(types.SimpleNamespace(project=str(path), device=0))
        return calls[:]

    return types.SimpleNamespace(data=data, model=model, project=project, add=add_quant,
                                 run=run, calls=calls, fail=fail, root=tmp_path)


def test_incremental_measurements_survive_metadata_updates_and_interruption(harness):
    h = harness
    quant = h.add("3")
    assert h.run() == ["BF16", "BF16", "3"]
    stamp = h.data.source_stamp(str(h.model))
    metadata = h.model / ".ezexl3_readme_meta.json"
    metadata.write_text('{"updated": true}')
    os.utime(metadata, (stamp + 100, stamp + 100))
    assert h.data.source_stamp(str(h.model)) == stamp
    assert h.run() == []
    h.add("4")
    h.fail.add("4")
    with pytest.raises(RuntimeError, match="interrupted"):
        h.run()
    assert h.calls == ["4"]
    h.fail.clear()
    assert h.run() == ["4"]
    weights = quant / "model.safetensors"
    os.utime(weights, (stamp + 200, stamp + 200))
    assert h.run() == ["3"]
    results = json.loads((h.root / "results.json").read_text())
    assert [r["label"] for r in results] == ["BF16", "Noise floor", "3", "4"]


def test_evicted_logits_only_rebuilt_for_pending_comparisons(harness):
    h = harness
    h.add("3")
    h.run()
    for path in (h.root / "cache/qbench").glob("logits_*"):
        shutil.rmtree(path)
    assert h.run() == []
    h.add("4")
    assert h.run() == ["BF16", "4"]
    # A surviving meta.json alone is not a complete logit cache.
    for path in (h.root / "cache/qbench").glob("logits_*/row_*"):
        path.unlink()
    h.add("5")
    assert h.run() == ["BF16", "5"]


def test_reference_change_invalidates_comparisons(harness):
    h = harness
    h.add("3")
    h.run()
    config = h.model / "config.json"
    stamp = h.data.source_stamp(str(h.model)) + 100
    os.utime(config, (stamp, stamp))
    assert h.run() == ["BF16", "BF16", "3"]


def test_atomic_results_keep_previous_checkpoint_on_write_failure(harness, monkeypatch):
    cache = harness.data.QCache(harness.project["logit_cache"])
    cache.save_results("test", {"ppl": 2})
    monkeypatch.setattr(harness.data.json, "dump", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        cache.save_results("test", {"ppl": 3})
    assert cache.load_results("test") == {"ppl": 2}


def test_enabling_combined_histogram_reuses_cached_per_token_measurements(harness):
    h = harness
    h.add("3")
    h.run()
    h.project["output"]["plot_kld_hist_combined"] = str(h.root / "combined.png")
    assert h.run() == []
    plot = sys.modules["qbench.plot"].plot_kld_hist_combined
    plot.assert_called_once()
    assert plot.call_args.args[0][0]["label"] == "3"
    assert plot.call_args.args[4] == str(h.root / "combined.png")
