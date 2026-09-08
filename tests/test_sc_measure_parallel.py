import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ezexl3 import sc_measure_parallel as parallel


def row(key, idx=0):
    return dict(key=key, idx=idx, levels=[dict(kld=0.25)])


def shard(path, metadata, index, count, rows, complete=True):
    data = dict(metadata, results=rows,
                worker=dict(index=index, count=count,
                            expected_keys=[r["key"] for r in rows], complete=complete))
    parallel.atomic_json(path, data)
    return data


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_n_workers_map_selected_devices_and_merge(tmp_path, monkeypatch, count):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(f"GPU-test-{i}" for i in range(count)))
    output = tmp_path / "measurement.json"
    calls = []
    metadata = parallel.measurement_metadata(["-m", "toy", "--shaped"])

    def popen(cmd, env):
        index = int(cmd[cmd.index("--worker-index") + 1])
        assert cmd[cmd.index("--worker-count") + 1] == str(count)
        assert cmd[cmd.index("-d") + 1] == "0"
        assert env["CUDA_VISIBLE_DEVICES"] == f"GPU-test-{index}"
        resume = Path(cmd[cmd.index("--resume-from") + 1])
        assert json.loads(resume.read_text())["results"] == []
        path = Path(cmd[cmd.index("-o") + 1])
        shard(path, metadata, index, count, [row(f"layer.{index}", index)])
        calls.append(cmd)
        return SimpleNamespace(poll=lambda: 0, wait=lambda timeout=None: 0)

    parallel.run_parallel(list(range(count)), output, ["-m", "toy", "--shaped"], popen=popen)
    result = json.loads(output.read_text())
    assert len(calls) == count
    assert result == dict(metadata, results=[row(f"layer.{i}", i) for i in range(count)])
    assert "worker" not in result  # optimizer continues to consume its original schema


@pytest.mark.parametrize("visible,device,expected", [(None, 4, "4"), ("2,5,7", 1, "5"),
                                                     ("GPU-abc,GPU-def", 0, "GPU-abc")])
def test_logical_gpu_mapping(visible, device, expected):
    env = {} if visible is None else {"CUDA_VISIBLE_DEVICES": visible}
    assert parallel.worker_environment(device, env)["CUDA_VISIBLE_DEVICES"] == expected


def test_invalid_visible_device_fails_before_launch():
    with pytest.raises(ValueError, match="outside CUDA_VISIBLE_DEVICES"):
        parallel.worker_environment(2, {"CUDA_VISIBLE_DEVICES": "4,5"})


def test_resume_with_changed_gpu_count_imports_old_shards(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    output = tmp_path / "measurement.json"
    directory = Path(str(output) + ".workers")
    directory.mkdir()
    metadata = parallel.measurement_metadata(["-m", "toy"])
    original = row("a")
    saved_in_shard = row("b", 1)
    parallel.atomic_json(output, dict(metadata, results=[original]))
    shard(directory / "worker-2-of-3.json", metadata, 2, 3, [saved_in_shard], complete=False)

    def popen(cmd, env):
        resume = json.loads(Path(cmd[cmd.index("--resume-from") + 1]).read_text())
        assert resume["results"] == [original, saved_in_shard]
        shard(Path(cmd[cmd.index("-o") + 1]), metadata, 0, 1,
              [original, saved_in_shard, row("c", 2)])
        return SimpleNamespace(poll=lambda: 0, wait=lambda timeout=None: 0)

    parallel.run_parallel([0], output, ["-m", "toy"], popen=popen)
    assert len(json.loads(output.read_text())["results"]) == 3


def test_failed_worker_terminates_peers_and_salvages_progress(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    output = tmp_path / "measurement.json"
    metadata = parallel.measurement_metadata(["-m", "toy"])
    peers = []

    def popen(cmd, env):
        index = int(cmd[cmd.index("--worker-index") + 1])
        shard(Path(cmd[cmd.index("-o") + 1]), metadata, index, 2,
              [row(f"layer.{index}", index)], complete=False)
        proc = SimpleNamespace(rc=1 if index == 0 else None, terminated=False)
        proc.poll = lambda: proc.rc
        proc.wait = lambda timeout=None: proc.rc

        def terminate():
            proc.terminated = True
            proc.rc = -15

        proc.terminate = terminate
        peers.append(proc)
        return proc

    with pytest.raises(RuntimeError, match="failed"):
        parallel.run_parallel([0, 1], output, ["-m", "toy"], popen=popen)
    assert peers[1].terminated
    assert len(json.loads(output.read_text())["results"]) == 2


def test_exit_zero_without_complete_output_is_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    output = tmp_path / "measurement.json"
    with pytest.raises(RuntimeError, match="without an output"):
        parallel.run_parallel([0], output, ["-m", "toy"],
                              popen=lambda *a, **k: SimpleNamespace(poll=lambda: 0, wait=lambda timeout=None: 0))


def test_incompatible_or_conflicting_shards_do_not_replace_checkpoint(tmp_path):
    output = tmp_path / "measurement.json"
    path = tmp_path / "shard.json"
    metadata = parallel.measurement_metadata(["-m", "toy"])
    parallel.atomic_json(output, dict(metadata, results=[row("a")]))
    original = output.read_bytes()
    shard(path, dict(metadata, rows=999), 0, 2, [row("b")])
    with pytest.raises(ValueError, match="settings differ"):
        parallel.merge_outputs(output, [path], metadata)
    assert output.read_bytes() == original
    shard(path, metadata, 0, 2, [dict(row("a"), idx=999)])
    with pytest.raises(ValueError, match="Conflicting"):
        parallel.merge_outputs(output, [path], metadata)
    assert output.read_bytes() == original


def test_overlapping_worker_assignments_are_rejected(tmp_path):
    metadata = parallel.measurement_metadata(["-m", "toy"])
    paths = [tmp_path / f"{i}.json" for i in range(2)]
    for index, path in enumerate(paths):
        shard(path, metadata, index, 2, [row("a")])
    with pytest.raises(RuntimeError, match="overlapping"):
        parallel.validate_completed_shards(paths, dict(metadata, results=[row("a")]))


def test_cancellation_retains_atomic_worker_outputs(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    output = tmp_path / "measurement.json"
    metadata = parallel.measurement_metadata(["-m", "toy"])
    terminated = []

    def popen(cmd, env):
        index = int(cmd[cmd.index("--worker-index") + 1])
        if index == 1:
            raise KeyboardInterrupt()
        shard(Path(cmd[cmd.index("-o") + 1]), metadata, index, 2, [row("saved")], complete=False)
        proc = SimpleNamespace(rc=None)
        proc.poll = lambda: proc.rc
        proc.wait = lambda timeout=None: proc.rc

        def terminate():
            terminated.append(index)
            proc.rc = -15

        proc.terminate = terminate
        return proc

    with pytest.raises(KeyboardInterrupt):
        parallel.run_parallel([0, 1], output, ["-m", "toy"], popen=popen)
    assert terminated == [0]
    assert json.loads(output.read_text())["results"] == [row("saved")]


def test_real_processes_run_concurrently(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    metadata = parallel.measurement_metadata(["-m", "toy"])
    script = tmp_path / "worker.py"
    script.write_text('''
import argparse, json, os, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--worker-index", type=int)
p.add_argument("--worker-count", type=int)
p.add_argument("-o")
a, _ = p.parse_known_args()
path = Path(a.o)
path.with_suffix(".ready").touch()
deadline = time.monotonic() + 10
while len(list(path.parent.glob("*.ready"))) < a.worker_count:
    if time.monotonic() > deadline:
        raise RuntimeError("Workers were not launched concurrently")
    time.sleep(0.02)
metadata = ''' + repr(metadata) + '''
assert os.environ["CUDA_VISIBLE_DEVICES"] == str(a.worker_index)
key = "layer." + str(a.worker_index)
metadata["results"] = [{"key": key, "idx": a.worker_index, "levels": [{"kld": 0.25}]}]
metadata["worker"] = dict(index=a.worker_index, count=a.worker_count,
                          expected_keys=[key], complete=True)
tmp = Path(str(path) + ".tmp")
tmp.write_text(json.dumps(metadata))
os.replace(tmp, path)
''')
    output = tmp_path / "measurement.json"
    parallel.run_parallel([0, 1, 2], output, ["-m", "toy"], script=str(script), poll_interval=0.02)
    assert len(json.loads(output.read_text())["results"]) == 3


def test_cli_preserves_draws_and_worker_options(monkeypatch):
    import sys
    calls = []
    monkeypatch.setattr(sys, "argv", ["coordinator", "--devices", "0,1,2", "-o", "out.json",
                                    "-m", "toy", "-d", "0", "-dr", "3", "--streaming"])
    monkeypatch.setattr(parallel.signal, "signal", lambda *args: None)
    monkeypatch.setattr(parallel, "run_parallel", lambda *args: calls.append(args))
    parallel.main()
    devices, output, forwarded = calls[0]
    assert devices == [0, 1, 2]
    assert output == "out.json"
    assert "--streaming" in forwarded
    assert parallel.measurement_metadata(forwarded)["draws"] == 3
    defaults = parallel.measurement_metadata(["-m", "toy", "-d", "0", "-t", "20"])
    assert defaults["draws"] == 1
    assert defaults["trace"] is None
