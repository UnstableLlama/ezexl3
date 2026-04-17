"""Wrapper around the vendored ``eval_perf.py`` that adds inner-loop
heartbeat output without touching the upstream-vendored script.

The vendored script's ``measure_prefill`` and ``measure_generate`` only
print a single line per context length — for long contexts on slow
quants those inner loops can stall the visible terminal for minutes.
We monkey-patch them with versions that periodically emit
``PERF_HEARTBEAT`` lines that ``ezexl3.evals._parse_perf_progress``
recognises and forwards to the UI as a single updating line.

This wrapper takes the same CLI as ``eval_perf.py`` and is invoked by
``ezexl3.evals.build_eval_cmd("perf", ...)`` instead of the vendored
file directly.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

from ezexl3.vendor import eval_perf


def _measure_prefill_with_heartbeat(args, model, cache, warmup=False):
    """Drop-in replacement for ``eval_perf.measure_prefill`` that emits
    ``PERF_HEARTBEAT`` lines after each chunk so the UI shows progress
    during long prefill runs.
    """
    chunk_size = args.chunk_size
    lengths = eval_perf.get_lengths(chunk_size if warmup else args.max_length)

    progress = 0
    results: dict[int, float] = {}
    max_progress = sum(lengths)
    with eval_perf.ProgressBar("Warmup" if warmup else "Prefill", max_progress) as pb:
        for length in lengths:
            eval_perf.cuda_sync_active()
            with eval_perf.Timer() as t:
                start, end = 0, length
                pre_time = 0.0
                if length >= chunk_size * 2:
                    pre_time = (length // 2) / results[length // 2]
                    start = length // 2
                chunks = [
                    (i, min(i + chunk_size, end))
                    for i in range(start, end, chunk_size)
                ]
                if not warmup:
                    sys.stdout.write(
                        f"PERF_HEARTBEAT prefill length={length} chunk=0/{len(chunks)}\n"
                    )
                    sys.stdout.flush()
                _t0 = time.monotonic()
                for ci, (cstart, cend) in enumerate(chunks):
                    params = {
                        "attn_mode": "flash_attn",
                        "cache": cache,
                        "past_len": cstart,
                        "batch_shape": (1, max(length, 256)),
                    }
                    if "recurrent_states" in model.caps and cstart > 0:
                        for v in eval_perf.faux_recurrent_states.values():
                            v.position = cstart
                        params.update({
                            "recurrent_states": eval_perf.faux_recurrent_states
                        })
                    model.prefill(eval_perf.cached_ids(cend - cstart), params)
                    if (
                        "recurrent_states" in params
                        and eval_perf.faux_recurrent_states is None
                    ):
                        eval_perf.faux_recurrent_states = params["recurrent_states"]
                    if not warmup and ci + 1 < len(chunks):
                        elapsed = time.monotonic() - _t0
                        tokens = cend
                        tps = tokens / elapsed if elapsed > 0 else 0.0
                        sys.stdout.write(
                            f"PERF_HEARTBEAT prefill length={length} "
                            f"chunk={ci + 1}/{len(chunks)} {tokens} tokens "
                            f"({tps:.2f} t/s)\n"
                        )
                        sys.stdout.flush()
                eval_perf.cuda_sync_active()

            results[length] = length / (pre_time + t.interval)
            if not warmup:
                print(
                    f"Length  {length: 6}: "
                    f"{eval_perf.col_green}{results[length]:10.2f}"
                    f"{eval_perf.col_default} tokens/s"
                )
            progress += length
            pb.update(progress)

    return results


def _measure_generate_with_heartbeat(args, model, cache, warmup=False):
    """Drop-in replacement for ``eval_perf.measure_generate`` that emits
    ``PERF_HEARTBEAT`` lines roughly once per second during the inner
    100-iteration forward loop.
    """
    chunk_size = args.chunk_size
    # Upstream shrinks past_len by 256 and pads batch_shape by 256 so the
    # 100-iteration forward past past_len fits in a cache sized at max_length.
    lengths = [0] + eval_perf.get_lengths(chunk_size if warmup else args.max_length - 256)
    progress = 0
    results: dict[int, float] = {}
    max_progress = len(lengths)
    hb_interval = 1.0
    with eval_perf.ProgressBar("Warmup" if warmup else "Generate", max_progress) as pb:
        for length in lengths:
            torch.cuda.synchronize()
            last_hb = 0.0
            if not warmup:
                sys.stdout.write(
                    f"PERF_HEARTBEAT gen length={length} 0/100\n"
                )
                sys.stdout.flush()
                last_hb = time.monotonic()
            t0 = time.monotonic()
            with eval_perf.Timer() as t:
                for i in range(100):
                    params = {
                        "attn_mode": "flash_attn",
                        "cache": cache,
                        "past_len": length,
                        "batch_shape": (1, max(length + 256, 256)),
                    }
                    if "recurrent_states" in model.caps and length > 0:
                        for v in eval_perf.faux_recurrent_states.values():
                            v.position = length
                        params.update({
                            "recurrent_states": eval_perf.faux_recurrent_states
                        })
                    logits = model.forward(eval_perf.cached_ids(1), params)
                    sample = torch.argmax(logits)
                    sample = sample.cpu()  # force sync
                    del logits
                    if not warmup:
                        now = time.monotonic()
                        if now - last_hb >= hb_interval and i + 1 < 100:
                            elapsed = now - t0
                            done = i + 1
                            tps = done / elapsed if elapsed > 0 else 0.0
                            sys.stdout.write(
                                f"PERF_HEARTBEAT gen length={length} "
                                f"{done}/100 ({tps:.2f} t/s)\n"
                            )
                            sys.stdout.flush()
                            last_hb = now
            results[length] = 100 / t.interval
            if not warmup:
                print(
                    f"Context {length: 6}: "
                    f"{eval_perf.col_green}{results[length]:10.2f}"
                    f"{eval_perf.col_default} tokens/s"
                )
            progress += 1
            pb.update(progress)

    return results


def main() -> None:
    # Patch in our heartbeat-emitting versions before main runs.
    eval_perf.measure_prefill = _measure_prefill_with_heartbeat
    eval_perf.measure_generate = _measure_generate_with_heartbeat

    parser = argparse.ArgumentParser()
    eval_perf.model_init.add_args(parser, default_cache_size=32768)
    parser.add_argument(
        "-max_length", "--max_length", type=int,
        help="Max context length to measure (default: 32768)", default=32768,
    )
    parser.add_argument(
        "-chunk_size", "--chunk_size", type=int,
        help="Max chunk size (default: 4096)", default=4096,
    )
    args = parser.parse_args()
    eval_perf.main(args)


if __name__ == "__main__":
    main()
