from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepoPlan:
    requested_integers: list[str]
    requested_optimizeds: list[str]
    quant_integer_queue: list[str]
    measure_queue: list[str]


@dataclass(frozen=True)
class MeasureRuntimeConfig:
    model_dir: str
    devices: list[int]
    db_path: str
    out_csv: str
    ppl_rows: int
    write_logs: bool = True
