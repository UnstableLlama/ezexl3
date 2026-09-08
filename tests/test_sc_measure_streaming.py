"""Exercise the actual measurement loop with CPU tensors and reloadable toy modules.

AST loading avoids importing exllamav3's CUDA extension on CPU-only test hosts.
Only its external dependencies are replaced; the measurement functions run unchanged.
"""
import ast
from collections import defaultdict
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import zlib

import pytest

torch = pytest.importorskip("torch")
SOURCE = Path(__file__).parents[1] / "ezexl3/vendor/sc_measure.py"


@pytest.fixture
def harness(monkeypatch):
    if not isinstance(torch, ModuleType):
        pytest.skip("Another test replaced torch globally; covered by the isolated subprocess test")
    tracker = SimpleNamespace(live=0, peak=0, full_loads=0, fail_load=False,
                              fail_forward=False, fail_perturbed=False,
                              finalized=0, models=[])

    class FP16:
        pass

    class Linear:
        def __init__(self, key, weight, qmap=None):
            self.key = key
            self.original = weight
            self.in_features, self.out_features = weight.shape
            self.qmap = qmap
            self.qbits_key = "bits"
            self.modules = []
            self.caps = {"prefer_cpu": True}
            self.inner = None

        def weights_numel(self):
            return self.original.numel()

        def can_defer_load(self):
            return True

        def load(self, device):
            assert self.inner is None
            self.inner = FP16()
            self.inner.weight = self.original.clone()
            tracker.live += 1
            tracker.peak = max(tracker.peak, tracker.live)

        def unload(self):
            if self.inner is not None:
                # Every perturbation must be restored even when a forward fails.
                torch.testing.assert_close(self.inner.weight, self.original, rtol=0, atol=0)
                tracker.live -= 1
                self.inner = None

        def prepare_for_device(self, x, params):
            return x

        def forward(self, x, params):
            if tracker.fail_forward and self.qmap:
                raise RuntimeError("injected forward failure")
            if tracker.fail_perturbed and not torch.equal(self.original, self.inner.weight):
                raise RuntimeError("injected perturbed forward failure")
            if "capture" in params and self.qmap:
                params["capture"].setdefault(self.qmap, {"k": self.in_features})
            # Deliberately mutate inputs to catch cache aliasing on prefer_cpu modules.
            x = x.half() if not x.is_floating_point() else x
            x.add_(0.125)
            return (x @ self.inner.weight).half()

    class Model:
        @classmethod
        def from_config(cls, config):
            model = cls()
            model.modules = [
                Linear("embed", torch.tensor([[0.4, 0.1], [0.2, 0.5]]).half()),
                Linear("layer.0", torch.tensor([[0.7, 0.2], [-0.2, 0.4]]).half(), "hidden"),
                Linear("head", torch.tensor([[0.4, -0.5, 0.3], [0.1, 0.6, -0.2]]).half(), "head"),
            ]
            tracker.models.append(model)
            return model

        def load(self, device):
            tracker.full_loads += 1
            for i, mod in enumerate(self.modules):
                mod.load(device)
                if i == 0 and tracker.fail_load:
                    raise torch.cuda.OutOfMemoryError("simulated load OOM")

        def unload(self):
            for mod in self.modules:
                mod.unload()

    stc = SimpleNamespace(begin_deferred_load=lambda: None,
                          end_deferred_load=lambda: None, abort_deferred_load=lambda: None,
                          close=lambda: None,
                          tensor_file_map={"toy.weight": "toy.safetensors"},
                          file_headers={"toy.safetensors": {
                              "toy.weight": {"shape": [14], "data_offsets": [0, 28]}}})
    config = SimpleNamespace(stc=stc, override_dynamic_seq_len=lambda n: None)

    def finalize(h_data, quant_args, verbose):
        tracker.finalized += 1
        k = h_data["k"]
        return False, torch.eye(k), torch.zeros(k, k), torch.ones(k), torch.ones(k)

    def kld(x, ref, vocab):
        a = torch.log_softmax(ref[..., :vocab].float(), -1)
        b = torch.log_softmax(x[..., :vocab].float(), -1)
        return (a.exp() * (a - b)).sum(-1)

    namespace = dict(torch=torch, contextmanager=contextmanager, defaultdict=defaultdict,
                     math=math, os=os, json=json, zlib=zlib, Linear=Linear, LinearFP16=FP16,
                     Model=Model, Config=SimpleNamespace(from_directory=lambda _: config),
                     Tokenizer=SimpleNamespace(from_config=lambda _: SimpleNamespace(actual_vocab_size=3)),
                     compute_kl_div=kld, finalize_capture_H=finalize, had_k=2,
                     get_hadamard_dt=lambda k, device, dtype, scale: torch.eye(k),
                     g_tensor_cache=SimpleNamespace(drop_all=lambda: None),
                     disk_lru_cache=lambda _: lambda f: f)
    tree = ast.parse(SOURCE.read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    namespace["get_test_tokens"] = lambda tokenizer, rows, length: torch.tensor(
        [[[1, 2], [2, 1]]] * rows, dtype=torch.int64)
    real_device = torch.device
    monkeypatch.setattr(torch, "device", lambda *args: real_device("cpu"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (100 * 1024**3, 100 * 1024**3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return namespace, tracker


def args(path, mode="streaming", shaped=False):
    return SimpleNamespace(model="toy", device=0, load_mode=mode, rows=2, length=2,
                           h_rows=2, shaped=shaped, trace=None, rfn_ref=None,
                           rfn="0.29,0.145", rfn_scale="1.0,0.5", draws=2,
                           out=str(path), top=2, worker_index=0, worker_count=1, resume_from=None)


@pytest.mark.parametrize("shaped", [False, True])
def test_streaming_matches_resident_and_resumes(harness, tmp_path, shaped):
    ns, tracker = harness
    resident = tmp_path / "resident.json"
    streamed = tmp_path / "streamed.json"
    ns["main"](args(resident, "resident", shaped))
    assert tracker.peak == 3
    tracker.peak = 0
    ns["main"](args(streamed, "streaming", shaped))
    assert tracker.peak == 1
    assert tracker.live == 0
    expected = json.loads(resident.read_text())
    actual = json.loads(streamed.read_text())
    assert actual == expected
    assert len(actual["results"]) == 2
    assert all(r["shaped"] == shaped for r in actual["results"])
    # Leave only the first tensor complete, then resume through the output head.
    actual["results"] = actual["results"][:1]
    streamed.write_text(json.dumps(actual))
    ns["main"](args(streamed, "streaming", shaped))
    assert json.loads(streamed.read_text()) == expected
    assert tracker.live == 0


@pytest.mark.parametrize("free_gib,expected_loads", [(1, 0), (100, 1)])
def test_auto_preflight(harness, monkeypatch, tmp_path, free_gib, expected_loads):
    ns, tracker = harness
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (free_gib * 1024**3, 100 * 1024**3))
    ns["main"](args(tmp_path / "auto.json", "auto"))
    assert tracker.full_loads == expected_loads
    assert tracker.live == 0


def test_auto_load_oom_falls_back(harness, tmp_path):
    ns, tracker = harness
    tracker.fail_load = True
    ns["main"](args(tmp_path / "fallback.json", "auto"))
    assert tracker.full_loads == 1
    assert tracker.peak == 1
    assert tracker.live == 0


def test_forced_resident_does_not_silently_fall_back(harness, tmp_path):
    ns, tracker = harness
    tracker.fail_load = True
    with pytest.raises(torch.cuda.OutOfMemoryError):
        ns["main"](args(tmp_path / "resident.json", "resident"))


def test_streaming_unloads_on_forward_failure(harness):
    ns, tracker = harness
    model = ns["Model"].from_config(None)
    config = ns["Config"].from_directory(None)
    runner = ns["ModuleRunner"](config, model.modules, "cpu", True)
    tracker.fail_forward = True
    with pytest.raises(RuntimeError, match="injected forward failure"):
        with runner.loaded(1) as mod:
            runner.rows(mod, [torch.ones(1, 2)])
    assert tracker.live == 0


def test_estimate_includes_shaping_and_cached_rows(harness):
    ns, _ = harness
    mods = ns["Model"].from_config(None).modules
    stc = ns["Config"].from_directory(None).stc
    estimate = ns["estimate_resident_bytes"]
    base = estimate(mods, 2, 1024, 128000, False, 0, stc)
    assert estimate(mods, 2, 1024, 128000, True, 64, stc) > base
    assert estimate(mods, 10, 1024, 128000, False, 0, stc) > base


def test_preflight_uses_headers_when_unloaded_norm_has_no_weight_count(harness):
    ns, tracker = harness

    class Norm:
        modules = []

        def weights_numel(self):
            return None  # exllamav3 RMSNorm before load()

    class Transformer:
        modules = [Norm()]

        def weights_numel(self):
            return sum(mod.weights_numel() for mod in self.modules)

    mod = Transformer()
    with pytest.raises(TypeError):
        mod.weights_numel()  # reproduces the user's original traceback
    stc = SimpleNamespace(
        tensor_file_map={"norm.weight": "shard"},
        file_headers={"shard": {"norm.weight": {"shape": [8192], "data_offsets": [0, 16384]},
                                "__metadata__": {"format": "pt"}, "_header_offset": 256}},
    )
    estimate = ns["estimate_resident_bytes"]
    required = estimate([mod], 2, 1024, 128000, False, 0, stc)
    assert required > 3 * 1024**3
    # FP8 storage still requires FP16 model memory; FP32 storage is conservative.
    stc.file_headers["shard"]["norm.weight"]["data_offsets"] = [0, 8192]
    assert estimate([mod], 2, 1024, 128000, False, 0, stc) == required
    stc.file_headers["shard"]["norm.weight"]["data_offsets"] = [0, 32768]
    assert estimate([mod], 2, 1024, 128000, False, 0, stc) > required
    assert tracker.full_loads == 0


@pytest.mark.parametrize("workers", [1, 2, 3, 5])
def test_partitioned_measurements_match_single_worker(harness, tmp_path, workers):
    ns, tracker = harness
    baseline = tmp_path / "baseline.json"
    ns["main"](args(baseline, shaped=True))
    expected = json.loads(baseline.read_text())
    # Simulate a partially completed single-GPU run before switching GPU count.
    resume = tmp_path / "resume.json"
    resume.write_text(json.dumps(dict(expected, results=expected["results"][:1])))
    combined = {}
    owners = set()
    for index in range(workers):
        path = tmp_path / f"worker-{index}.json"
        options = args(path, shaped=True)
        options.worker_count = workers
        options.worker_index = index
        options.resume_from = str(resume)
        ns["main"](options)
        result = json.loads(path.read_text())
        inventory = set(result["worker"]["expected_keys"])
        assert not owners & inventory
        owners |= inventory
        assert result["worker"]["complete"]
        assert {r["key"] for r in result["results"]} == inventory
        combined.update({r["key"]: r for r in result["results"]})
    assert combined == {r["key"]: r for r in expected["results"]}
    assert tracker.live == 0
    assert tracker.peak == 1


def test_module_assignment_balances_suffix_replay_cost(harness):
    ns, _ = harness
    targets = {i: [object()] * 4 for i in range(24)}
    assign = ns["assign_measurement_modules"]
    owners = assign(targets, 24, 4)
    loads = [sum(len(targets[i]) * (24 - i) for i in targets if owners[i] == w)
             for w in range(4)]
    assert len(owners) == len(targets)
    assert max(loads) - min(loads) <= 4 * 24
    assert owners == assign(dict(reversed(list(targets.items()))), 24, 4)


def test_perturbation_restored_before_unload_on_failure(harness, tmp_path):
    ns, tracker = harness
    tracker.fail_perturbed = True
    with pytest.raises(RuntimeError, match="injected perturbed forward failure"):
        ns["main"](args(tmp_path / "failure.json"))
    assert tracker.live == 0  # unload also asserts exact weight restoration


@pytest.mark.parametrize("flags,mode", [([], "auto"), (["--streaming"], "streaming"),
                                       (["--no-streaming"], "resident"),
                                       (["--load-mode", "resident"], "resident")])
def test_loading_cli_defaults_and_overrides(monkeypatch, flags, mode):
    import argparse
    import sys
    tree = ast.parse(SOURCE.read_text())
    entry = tree.body[-1]
    assert isinstance(entry, ast.If)
    captured = []
    monkeypatch.setattr(sys, "argv", ["sc_measure.py", "-m", "toy"] + flags)
    exec(compile(ast.Module(body=entry.body, type_ignores=[]), str(SOURCE), "exec"),
         {"argparse": argparse, "main": captured.append})
    assert captured[0].load_mode == mode


def test_isolated_when_other_tests_replace_torch():
    # Existing chat tests replace sys.modules['torch'] during collection. Run the
    # numerical checks in a fresh interpreter in that case, without changing their mocks.
    if isinstance(torch, ModuleType):
        return
    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(Path(__file__).resolve()),
         "-k", "not isolated_when_other_tests_replace_torch"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
