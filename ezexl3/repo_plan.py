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
    bpws: List[str], opt_bpws: Optional[set] = None,
) -> Dict[str, List[str]]:
    ints, fracs = _split_integer_optimized_bpws(bpws)
    opt_bpws = opt_bpws or set()

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

    requested_ints = _dedupe_preserve_order(ints)
    requested_fracs = _dedupe_preserve_order(fracs)
    # Standard fractionals go into the quant queue alongside integers
    quant_queue = _dedupe_preserve_order(requested_ints + standard_fracs + required_neighbors)
    measure_targets = _dedupe_preserve_order(quant_queue + optimized_fracs)

    return {
        "requested_integers": requested_ints,
        "requested_optimizeds": optimized_fracs,
        "quant_integer_queue": quant_queue,
        "measure_queue": measure_targets,
    }
