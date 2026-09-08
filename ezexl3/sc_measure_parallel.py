"""GPU-free coordinator for independent self-calibration measurement workers.

Workers publish atomic JSON shards. Only this process writes the combined output;
shards survive cancellation and can be imported with a different GPU count later.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


META_KEYS = ("model", "rows", "length", "draws", "mode", "trace", "h_rows",
             "rfn_ref", "rfn_scale", "rfn")


def measurement_metadata(argv):
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # Register short options even when they don't affect result compatibility:
    # argparse otherwise treats -d as a prefix of -dr, and -t as a prefix of -tr.
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("-t", "--top", type=int, default=15)
    parser.add_argument("-o", "--out")
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-r", "--rows", type=int, default=10)
    parser.add_argument("-l", "--length", type=int, default=1024)
    parser.add_argument("-dr", "--draws", type=int, default=1)
    parser.add_argument("-sh", "--shaped", action="store_true")
    parser.add_argument("-hr", "--h_rows", type=int, default=64)
    parser.add_argument("-tr", "--trace")
    parser.add_argument("-rr", "--rfn_ref")
    parser.add_argument("-rs", "--rfn_scale", default="1.0,0.5")
    parser.add_argument("-rfn", "--rfn", default="0.29,0.145")
    args, _ = parser.parse_known_args(argv)
    return dict(model=args.model, rows=args.rows, length=args.length, draws=args.draws,
                mode="shaped" if args.shaped else "iid", trace=args.trace,
                h_rows=args.h_rows if args.shaped else None, rfn_ref=args.rfn_ref,
                rfn_scale=args.rfn_scale if args.rfn_ref else None,
                rfn=None if args.rfn_ref else args.rfn)


def atomic_json(path, data):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def merge_outputs(output, shard_paths, metadata):
    """Validate all inputs before replacing the checkpoint; never silently drop data."""
    output = Path(output)
    results = {}
    previous = None
    for path in [output, *shard_paths]:
        if not path.exists():
            continue
        with path.open() as f:
            data = json.load(f)
        if any(data.get(key) != metadata[key] for key in META_KEYS):
            raise ValueError(f"Measurement settings differ in {path}; use a separate output "
                             "for a different model or calibration configuration")
        if path == output:
            previous = data
        for row in data["results"]:
            key = row["key"]
            if key in results and results[key] != row:
                raise ValueError(f"Conflicting measurements for {key} in {path}")
            results[key] = row
    merged = dict(metadata, results=sorted(results.values(), key=lambda r: (r["idx"], r["key"])))
    if previous != merged:
        atomic_json(output, merged)
    return merged


def validate_completed_shards(paths, merged):
    expected = set()
    for index, path in enumerate(paths):
        if not path.exists():
            raise RuntimeError(f"Worker {index} exited without an output: {path}")
        with path.open() as f:
            data = json.load(f)
        worker = data.get("worker", {})
        keys = set(worker.get("expected_keys", []))
        result_keys = [row["key"] for row in data["results"]]
        if (worker.get("index") != index or worker.get("count") != len(paths)
                or not worker.get("complete") or set(result_keys) != keys
                or len(result_keys) != len(keys)):
            raise RuntimeError(f"Incomplete or invalid measurement shard: {path}")
        if expected & keys:
            raise RuntimeError(f"Workers have overlapping assignments: {path}")
        expected.update(keys)
    if {row["key"] for row in merged["results"]} != expected:
        raise RuntimeError("Combined measurements do not match the workers' target inventory")


def worker_environment(device, environ):
    """Resolve a logical device through the caller's visibility mask, including UUIDs."""
    env = dict(environ)
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        choices = [part.strip() for part in visible.split(",") if part.strip()]
        if device >= len(choices):
            raise ValueError(f"Device {device} is outside CUDA_VISIBLE_DEVICES={visible!r}")
        selected = choices[device]
    else:
        selected = str(device)
    env["CUDA_VISIBLE_DEVICES"] = selected
    env["PYTHONUNBUFFERED"] = "1"
    return env


def run_parallel(devices, output, forwarded, *, script=None, executable=sys.executable,
                 popen=subprocess.Popen, poll_interval=0.5):
    if not devices or any(d < 0 for d in devices) or len(set(devices)) != len(devices):
        raise ValueError("Select one or more distinct, non-negative GPU indices")
    if any(arg.split("=", 1)[0] in ("--worker-index", "--worker-count", "--resume-from")
           for arg in forwarded):
        raise ValueError("Worker assignment flags are managed by the coordinator")
    envs = [worker_environment(d, os.environ) for d in devices]
    if len({env["CUDA_VISIBLE_DEVICES"] for env in envs}) != len(envs):
        raise ValueError("Selected logical devices resolve to duplicate GPUs")
    output = Path(output).absolute()
    directory = Path(str(output) + ".workers")
    directory.mkdir(parents=True, exist_ok=True)
    metadata = measurement_metadata(forwarded)
    # Include every previous partition, not just today's GPU count.
    old_paths = sorted(directory.glob("worker-*-of-*.json"))
    merge_outputs(output, old_paths, metadata)
    count = len(devices)
    paths = [directory / f"worker-{i}-of-{count}.json" for i in range(count)]
    # Invalidate old completion flags before launching, while retaining all results.
    # This prevents an early-exiting worker from being mistaken for a successful run.
    for path in paths:
        if path.exists():
            with path.open() as f:
                data = json.load(f)
            data["worker"]["complete"] = False
            atomic_json(path, data)
    script = script or str(Path(__file__).parent / "vendor" / "sc_measure.py")
    print(f" -- Sensitivity measurement: {count} GPU worker(s); each keeps its own "
          "reference and activation caches in RAM/VRAM", flush=True)
    processes = []
    signatures = None
    all_paths = sorted(set(old_paths + paths))

    def merge_progress():
        nonlocal signatures
        current = [(p, p.stat().st_mtime_ns, p.stat().st_size)
                   for p in all_paths if p.exists()]
        if current != signatures:
            merge_outputs(output, all_paths, metadata)
            signatures = current

    try:
        for index, env in enumerate(envs):
            cmd = [executable, script, *forwarded, "-d", "0", "-o", str(paths[index]),
                   "--worker-index", str(index), "--worker-count", str(count),
                   "--resume-from", str(output)]
            print(f" -- Worker {index + 1}/{count}: GPU {env['CUDA_VISIBLE_DEVICES']}", flush=True)
            # Inherit stdout/stderr so the repo's existing log/SSE capture still works.
            processes.append(popen(cmd, env=env))
        while True:
            statuses = [proc.poll() for proc in processes]
            merge_progress()
            failed = [i for i, rc in enumerate(statuses) if rc is not None and rc != 0]
            if failed:
                raise RuntimeError(f"Sensitivity worker(s) {failed} failed; completed tensors "
                                   f"are retained in {output}")
            if all(rc is not None for rc in statuses):
                break
            time.sleep(poll_interval)
        merged = merge_outputs(output, all_paths, metadata)
        validate_completed_shards(paths, merged)
        print(f" -- Combined {len(merged['results'])} tensor measurements into {output}", flush=True)
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        # Atomic worker outputs remain readable even after forced termination.
        merge_progress()


def main():
    # Let dashboard/process termination run the same worker cleanup as Ctrl-C.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(allow_abbrev=False,
                                     description="Measure self-calibration sensitivity across N GPUs")
    parser.add_argument("--devices", required=True, help="Comma-separated logical GPU indices")
    parser.add_argument("-o", "--out", required=True)
    args, forwarded = parser.parse_known_args()
    run_parallel([int(d.strip()) for d in args.devices.split(",")], args.out, forwarded)


if __name__ == "__main__":
    main()
