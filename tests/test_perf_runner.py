"""Check heartbeat wrappers against upstream using CPU tensors and fake timing."""
import argparse
import ast
import inspect
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from ezexl3.evals import _extract_perf_result, _parse_perf_progress, extract_perf_detail

ROOT = Path(__file__).resolve().parents[1] / "ezexl3"


class Timer:
    interval = 1.0

    def __init__(self, *args):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, *args):
        pass


def functions(path, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("warmup", [False, True])
@pytest.mark.parametrize("spec_dec", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "generate"])
def test_heartbeat_preserves_upstream_measurements(warmup, spec_dec, phase, capsys):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None), argmax=torch.argmax)
    upstream = functions(ROOT / "vendor/eval_perf.py", dict(
        torch=fake_torch, Timer=Timer, ProgressBar=Timer,
        cuda_sync_active=lambda: None, col_gray="", col_green="", col_default="",
        _workload_ids=torch.arange(4096).reshape(1, -1),
    ))
    wrapper = functions(ROOT / "perf_runner.py", dict(
        eval_perf=SimpleNamespace(**upstream), torch=fake_torch, sys=sys, time=time,
    ))
    args = SimpleNamespace(chunk_size=256, max_length=512, short_prefill=False, spec_dec=spec_dec)

    def run(fn):
        calls = []
        states = []

        def forward(ids, params):
            calls.append((ids.tolist(), params["past_len"], params["batch_shape"]))
            return torch.ones(1, 2)

        def state(start):
            result = Mock()
            states.append(result)
            return result

        model = SimpleNamespace(caps={"recurrent_states": True}, forward=forward, prefill=forward)
        result = fn(args, model, SimpleNamespace(get_test_state=state), warmup=warmup)
        for s in states:
            s.free.assert_called_once()
        return result, calls

    expected = run(upstream[f"measure_{phase}"])
    capsys.readouterr()
    assert run(wrapper[f"_measure_{phase}_with_heartbeat"]) == expected
    output = capsys.readouterr().out
    if not warmup:
        assert "PERF_HEARTBEAT" in output
        if phase == "generate" and not spec_dec:
            assert extract_perf_detail(output)["generation"]


@pytest.mark.parametrize("modern", [False, True])
def test_cli_accepts_upstream_flags_without_duplicate_chunk_option(modern, monkeypatch):
    def legacy(parser, **kwargs):
        pass

    def current(parser, default_chunk_size=4096, **kwargs):
        parser.add_argument("-chunk_size", "--chunk_size", type=int, default=default_chunk_size)

    upstream = SimpleNamespace(model_init=SimpleNamespace(add_args=current if modern else legacy), main=Mock())
    wrapper = functions(ROOT / "perf_runner.py", dict(eval_perf=upstream, argparse=argparse, inspect=inspect))
    monkeypatch.setattr(sys, "argv", ["perf", "--chunk_size", "512", "--skip_gen", "--spec_dec"])
    wrapper["main"]()
    args = upstream.main.call_args.args[0]
    assert (args.chunk_size, args.skip_gen, args.spec_dec) == (512, True, True)


def test_new_perf_rows_reach_summary_detail_and_progress():
    output = "Length 512: 900.00 tokens/s\nContext 0: S=\x1b[37;1m1 \x1b[32;1m120.00\x1b[0m tokens/s\n"
    assert _extract_perf_result(output) == {"perf_prefill_tps": "900.00", "perf_gen_tps": "120.00"}
    assert extract_perf_detail(output)["generation"] == [(0, 120.0)]
    assert _parse_perf_progress(output.splitlines()[1]) == "gen @0: 120.00 t/s"
    # Speculative benchmark iterations are not tokens per second.
    assert extract_perf_detail("Context 0: S=1 120.00 it/s, S=2 90.00 it/s")["generation"] == []
