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
import inspect
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
    if args.short_prefill:
        lengths = list(range(lengths[0])) + lengths

    ids_offset = 0 if warmup else args.max_length
    is_recurrent = model.caps.get("recurrent_states", False)
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
                recurrent = [cache.get_test_state(start)] if is_recurrent else None
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
                        "recurrent_states": recurrent,
                    }
                    model.prefill(eval_perf.workload_ids(cstart + ids_offset, cend - cstart), params)
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
                if is_recurrent:
                    recurrent[0].free()

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


def _measure_generate_with_heartbeat(args, model, cache, warmup = False):
    chunk_size = args.chunk_size
    lengths = [0] + eval_perf.get_lengths(chunk_size if warmup else args.max_length - 256)

    ids_offset = args.max_length * 2 if warmup else args.max_length * 3
    is_recurrent = model.caps.get("recurrent_states", False)
    progress = 0
    results = {}
    seqlens = [1, 2, 3, 4] if args.spec_dec else [1]
    unit = "it" if args.spec_dec else "tokens"
    max_progress = len(lengths)
    with (eval_perf.ProgressBar("Warmup" if warmup else "Generate", max_progress) as pb):
        for length in lengths:
            for seqlen in seqlens:
                recurrent = [cache.get_test_state(length)] if is_recurrent else None
                torch.cuda.synchronize()
                last_hb = time.monotonic()
                if not warmup:
                    print(f"PERF_HEARTBEAT gen length={length} S={seqlen} 0/{100 // seqlen}", flush=True)
                with eval_perf.Timer() as t:
                    for i in range(100 // seqlen):
                        params = {
                            "attn_mode": "flash_attn",
                            "cache": cache,
                            "past_len": length + i * seqlen,
                            "batch_shape": (1, max(length + 256, 256)),
                            "recurrent_states": recurrent
                        }
                        logits = model.forward(eval_perf.workload_ids(ids_offset + length + i, seqlen), params)
                        sample = torch.argmax(logits)
                        sample = sample.cpu()  # force sync
                        del logits
                        if not warmup and time.monotonic() - last_hb >= 1.0:
                            print(f"PERF_HEARTBEAT gen length={length} S={seqlen} {i + 1}/{100 // seqlen}", flush=True)
                            last_hb = time.monotonic()
                if is_recurrent:
                    recurrent[0].free()
                results[seqlen, length] = (100 // seqlen) / t.interval

            if not warmup:
                print(
                    f"Context {length: 6}: " +
                    ",   ".join([
                        f"S={eval_perf.col_gray}{seqlen} {eval_perf.col_green}{results[seqlen, length]:10.2f}{eval_perf.col_default} {unit}/s"
                        for seqlen in seqlens
                    ])
                )

            progress += 1
            pb.update(progress)

    return results


def main() -> None:
    # Patch in our heartbeat-emitting versions before main runs.
    eval_perf.measure_prefill = _measure_prefill_with_heartbeat
    eval_perf.measure_generate = _measure_generate_with_heartbeat

    parser = argparse.ArgumentParser(allow_abbrev=False)
    init_options = {}
    if "default_chunk_size" in inspect.signature(eval_perf.model_init.add_args).parameters:
        init_options["default_chunk_size"] = 4096
    eval_perf.model_init.add_args(
        parser,
        default_cache_size=32768,
        default_autosplit_max_batch_size=1,
        **init_options,
    )
    if not init_options:
        parser.add_argument("-chunk_size", "--chunk_size", type=int, default=4096)
    parser.add_argument(
        "-max_length", "--max_length", type=int,
        help="Max context length to measure (default: 32768)", default=32768,
    )
    parser.add_argument(
        "-spf", "--skip_prefill", action="store_true",
        help="Skip measuring prefill speed",
    )
    parser.add_argument(
        "-swu", "--skip_warmup", action="store_true",
        help="Skip warmup passes",
    )
    parser.add_argument(
        "-short", "--short_prefill", action="store_true",
        help="Test short-prefill/batch throughput",
    )
    parser.add_argument("-sg", "--skip_gen", action="store_true", help="Skip generation speed")
    parser.add_argument("-sd", "--spec_dec", action="store_true", help="Test spec-decode sequence lengths 1..4")
    args = parser.parse_args()
    eval_perf.main(args)


if __name__ == "__main__":
    main()
