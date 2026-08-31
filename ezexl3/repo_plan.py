from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


def _normalize_bpw_str(raw: str) -> str:
    token = str(raw).strip()
    if not token:
        raise ValueError("Empty BPW value provided")
    try:
        numeric = float(token)
    except ValueError as e:
        raise ValueError(f"Invalid BPW value '{raw}'") from e
    if numeric <= 0:
        raise ValueError(f"BPW values must be > 0, got '{raw}'")

    if "." not in token:
        return str(int(numeric)) if numeric.is_integer() else token

    trimmed = token.rstrip("0").rstrip(".")
    if not trimmed:
        return str(int(numeric)) if numeric.is_integer() else token
    if "." not in trimmed and numeric.is_integer():
        return str(int(numeric))
    return trimmed


def _split_integer_optimized_bpws(bpws: List[str]) -> Tuple[List[str], List[str]]:
    integer_bpws: List[str] = []
    optimized_bpws: List[str] = []

    for raw in bpws:
        normalized = _normalize_bpw_str(raw)
        value = float(normalized)
        if math.isclose(value, round(value), abs_tol=1e-9):
            integer_bpws.append(str(int(round(value))))
        else:
            optimized_bpws.append(normalized)
    return integer_bpws, optimized_bpws


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for it in items:
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def _plan_repo_bpws(
    bpws: List[str], opt_bpws: Optional[set] = None, sc_bpws: Optional[set] = None,
) -> Dict[str, List[str]]:
    ints, fracs = _split_integer_optimized_bpws(bpws)
    opt_bpws = opt_bpws or set()
    sc_set = {_normalize_bpw_str(b) for b in (sc_bpws or set())}

    conflict = sc_set & {_normalize_bpw_str(b) for b in opt_bpws}
    if conflict:
        raise ValueError(
            "BPW(s) painted with both -sc and -opt: "
            + ", ".join(sorted(conflict, key=float))
            + " — a BPW is built by one pipeline or the other, not both"
        )

    # -sc painted BPWs (integer or fractional) leave the plain quant queue
    # entirely; they are built by the self-calibration pipeline instead.
    selfcal_targets = [b for b in _dedupe_preserve_order(ints + fracs) if b in sc_set]
    ints = [b for b in ints if b not in sc_set]
    fracs = [b for b in fracs if b not in sc_set]

    # Only fractionals painted with -opt need the optimization pipeline;
    # the rest are quantized directly like integer BPWs.
    optimized_fracs = [f for f in fracs if f in opt_bpws]
    standard_fracs = [f for f in fracs if f not in opt_bpws]

    required_neighbors: List[str] = []
    for frac in optimized_fracs:
        frac_val = float(frac)
        low = math.floor(frac_val)
        high = math.ceil(frac_val)
        required_neighbors.extend([str(low), str(high)])

    # An -opt target needs its plain integer neighbors at <model>/<bpw>; an
    # -sc paint on such a neighbor would build a different quant at the same
    # path. Refuse the ambiguity instead of silently picking one.
    neighbor_conflict = sc_set & set(required_neighbors)
    if neighbor_conflict:
        raise ValueError(
            "BPW(s) painted with -sc are required as plain integer donors "
            "for -opt targets: " + ", ".join(sorted(neighbor_conflict, key=float))
        )

    requested_ints = _dedupe_preserve_order(ints)
    requested_fracs = _dedupe_preserve_order(fracs)
    # Quant queue: integers + integer neighbors of -opt targets + standard
    # (non-opt) fractionals, sorted numerically so the run progresses
    # monotonically through the BPW range instead of batching fractionals
    # at the tail.
    quant_queue = sorted(
        _dedupe_preserve_order(requested_ints + standard_fracs + required_neighbors),
        key=float,
    )
    # Measure queue: everything in the quant queue plus the -opt and -sc
    # targets, also in numeric order. Both can be interleaved here because
    # the measure stage always runs after their pipelines have produced the
    # corresponding BPW directories.
    measure_targets = sorted(
        _dedupe_preserve_order(quant_queue + optimized_fracs + selfcal_targets),
        key=float,
    )

    return {
        "requested_integers": requested_ints,
        "requested_optimizeds": optimized_fracs,
        "requested_selfcal": selfcal_targets,
        "quant_integer_queue": quant_queue,
        "measure_queue": measure_targets,
    }
