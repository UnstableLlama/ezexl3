from __future__ import annotations

import csv
import html
import os
from dataclasses import dataclass
from typing import List


@dataclass
class PlotRow:
    bpw: float
    kld: float
    ppl: float
    gib: float


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pad(lo: float, hi: float, frac: float = 0.10) -> tuple[float, float]:
    if hi == lo:
        return lo - 1.0, hi + 1.0
    delta = (hi - lo) * frac
    return lo - delta, hi + delta


def _kld_value(row: dict) -> float:
    if "K/L Div" in row:
        return _as_float(row.get("K/L Div"), 0.0)
    if "KL Div" in row:
        return _as_float(row.get("KL Div"), 0.0)
    return 0.0


def load_plot_rows(csv_path: str) -> List[PlotRow]:
    out: List[PlotRow] = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            weights = str(row.get("weights", "")).strip().lower()
            if weights == "bf16":
                continue

            bpw = _as_float(row.get("weights"), float("nan"))
            if bpw != bpw:
                continue

            out.append(
                PlotRow(
                    bpw=bpw,
                    kld=_kld_value(row),
                    ppl=_as_float(row.get("PPL r-100"), 0.0),
                    gib=_as_float(row.get("GiB"), 0.0),
                )
            )

    out.sort(key=lambda r: r.bpw)
    return out


def generate_iceblink_svg(csv_path: str, out_svg: str, title: str) -> str:
    rows = load_plot_rows(csv_path)
    if len(rows) < 2:
        raise ValueError("Need at least 2 non-bf16 rows to draw graph")

    width, height = 1365, 768
    left, right = 105, width - 95
    top, bottom = 90, height - 105

    bpw_vals = [r.bpw for r in rows]
    kld_vals = [r.kld for r in rows]
    ppl_vals = [r.ppl for r in rows]
    gib_vals = [r.gib for r in rows]

    xmin, xmax = min(bpw_vals), max(bpw_vals)
    kmin, kmax = _pad(min(kld_vals), max(kld_vals), 0.10)
    pmin, pmax = _pad(min(ppl_vals), max(ppl_vals), 0.10)

    def xmap(x: float) -> float:
        if xmax == xmin:
            return (left + right) / 2.0
        return left + (x - xmin) * (right - left) / (xmax - xmin)

    def ymap_left(y: float) -> float:
        if kmax == kmin:
            return (top + bottom) / 2.0
        return bottom - (y - kmin) * (bottom - top) / (kmax - kmin)

    def ymap_right(y: float) -> float:
        if pmax == pmin:
            return (top + bottom) / 2.0
        return bottom - (y - pmin) * (bottom - top) / (pmax - pmin)

    def points_str(xs: list[float], ys: list[float], yfn) -> str:
        return " ".join(f"{xmap(x):.2f},{yfn(y):.2f}" for x, y in zip(xs, ys))

    k_points = points_str(bpw_vals, kld_vals, ymap_left)
    p_points = points_str(bpw_vals, ppl_vals, ymap_right)

    grid_lines = []
    n_h = 6
    n_v = max(3, len(rows))
    for i in range(n_h + 1):
        y = top + (bottom - top) * i / n_h
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" '
            'stroke="#FFFFFF" stroke-opacity="0.25" stroke-dasharray="5 6"/>'
        )
    for i in range(n_v + 1):
        x = left + (right - left) * i / n_v
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}" '
            'stroke="#FFFFFF" stroke-opacity="0.18" stroke-dasharray="5 6"/>'
        )

    top_labels = []
    for row in rows:
        top_labels.append(
            f'<text x="{xmap(row.bpw):.2f}" y="{top - 20}" fill="#FFFFFF" '
            f'font-size="12" text-anchor="middle">{row.gib:.2f}</text>'
        )

    markers = []
    for row in rows:
        x = xmap(row.bpw)
        yk = ymap_left(row.kld)
        yp = ymap_right(row.ppl)
        markers.append(f'<circle cx="{x:.2f}" cy="{yk:.2f}" r="6" fill="#00E5FF"/>')
        markers.append(
            f'<rect x="{x-5:.2f}" y="{yp-5:.2f}" width="10" height="10" fill="#FF00FF"/>'
        )
        markers.append(
            f'<text x="{x:.2f}" y="{yk-16:.2f}" fill="#FFFFFF" font-size="24" text-anchor="middle">✓</text>'
        )

    escaped_title = html.escape(title)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#000000"/>
  {''.join(grid_lines)}
  <rect x="{left}" y="{top}" width="{right-left}" height="{bottom-top}" fill="none" stroke="#FFFFFF" stroke-width="1.2"/>
  <polyline points="{k_points}" fill="none" stroke="#00E5FF" stroke-width="3"/>
  <polyline points="{p_points}" fill="none" stroke="#FF00FF" stroke-width="3"/>
  {''.join(markers)}
  {''.join(top_labels)}

  <text x="{(left+right)/2:.2f}" y="52" fill="#00E5FF" font-size="34" font-weight="700" text-anchor="middle">{escaped_title}</text>
  <text x="{(left+right)/2:.2f}" y="{height-38}" fill="#FFFFFF" font-size="22" text-anchor="middle">Bits per Weight (BPW)</text>
  <text x="{(left+right)/2:.2f}" y="32" fill="#FFFFFF" font-size="22" text-anchor="middle">Model Size (GiB)</text>
  <text x="26" y="{(top+bottom)/2:.2f}" fill="#00E5FF" font-size="22" transform="rotate(-90, 26, {(top+bottom)/2:.2f})" text-anchor="middle">KL Div</text>
  <text x="{width-26}" y="{(top+bottom)/2:.2f}" fill="#FF00FF" font-size="22" transform="rotate(90, {width-26}, {(top+bottom)/2:.2f})" text-anchor="middle">PPL</text>
</svg>
'''

    os.makedirs(os.path.dirname(out_svg) or ".", exist_ok=True)
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)
    return out_svg

