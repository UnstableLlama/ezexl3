from __future__ import annotations

import argparse
import math
import os
from typing import Any, Sequence, Tuple



def _load_plot_libs() -> tuple[Any, Any, Any]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Graph generation requires matplotlib, numpy, and pandas to be installed."
        ) from exc
    return plt, np, pd


def load_series(
    csv_path: str,
    weights_col: str = "weights",
    kld_col: str = "KL Div",
    ppl_prefix: str = "PPL",
    gib_col: str = "GiB",
    drop_bf16: bool = True,
) -> Tuple[Any, Any, Any, Any, str]:
    _, _, pd = _load_plot_libs()
    df = pd.read_csv(csv_path)

    if drop_bf16 and weights_col in df.columns:
        df = df[df[weights_col].astype(str).str.lower() != "bf16"].copy()

    df["BPW"] = pd.to_numeric(df[weights_col], errors="coerce")
    df = df.dropna(subset=["BPW"]).sort_values("BPW")

    ppl_cols = [c for c in df.columns if c.strip().lower().startswith(ppl_prefix.lower())]
    if not ppl_cols:
        raise ValueError(f"No PPL column found (expected a column starting with '{ppl_prefix}').")
    ppl_col = ppl_cols[0]

    bpw = df["BPW"].to_numpy()
    kld = pd.to_numeric(df[kld_col], errors="coerce").to_numpy()
    ppl = pd.to_numeric(df[ppl_col], errors="coerce").to_numpy()
    gib = pd.to_numeric(df[gib_col], errors="coerce").to_numpy()

    return bpw, kld, ppl, gib, ppl_col


def pad(lo: float, hi: float, frac: float = 0.06) -> tuple[float, float]:
    if hi == lo:
        return lo - 1, hi + 1
    d = (hi - lo) * frac
    return lo - d, hi + d


def _top_axis_ticks_and_labels(gib_s: Sequence[float]) -> tuple[list[int], list[str]]:
    gmin, gmax = float(min(gib_s)), float(max(gib_s))
    start = math.floor(gmin)
    end = math.ceil(gmax)
    ticks = list(range(start, end + 1))
    if len(ticks) < 2:
        return ticks, [str(t) for t in ticks]

    # Hide the rightmost label due to uneven final spacing after interpolation.
    labels = [str(t) for t in ticks]
    labels[-1] = ""
    return ticks, labels


def make_plot(bpw, kld, ppl, gib, title, outfile, add_checks=True):
    plt, np, _ = _load_plot_libs()
    cyan = "#00E5FF"
    magenta = "#FF00FF"
    white = "#FFFFFF"
    bg = "#000000"

    fig, ax = plt.subplots(figsize=(13.65, 7.68), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    l1, = ax.plot(
        bpw,
        kld,
        color=cyan,
        linewidth=3.0,
        marker="o",
        markersize=8,
        markerfacecolor=cyan,
        markeredgecolor=cyan,
        label="KL Div",
    )

    ax.set_xlabel("Bits per Weight (BPW)", color=white, fontsize=16, labelpad=12)
    ax.set_ylabel("KL Div", color=cyan, fontsize=16, labelpad=14)
    ax.tick_params(axis="x", colors=white, labelsize=13, length=6, width=1.2)
    ax.tick_params(axis="y", colors=cyan, labelsize=13, length=6, width=1.2)

    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth=1.0, alpha=0.45, color=white)
    ax.grid(True, which="minor", linestyle="--", linewidth=0.7, alpha=0.25, color=white)

    for s in ax.spines.values():
        s.set_color(white)
        s.set_linewidth(1.2)

    axr = ax.twinx()
    axr.set_facecolor("none")
    l2, = axr.plot(
        bpw,
        ppl,
        color=magenta,
        linewidth=3.0,
        marker="s",
        markersize=8,
        markerfacecolor=magenta,
        markeredgecolor=magenta,
        label="PPL",
    )
    axr.set_ylabel("PPL", color=magenta, fontsize=16, labelpad=14)
    axr.tick_params(axis="y", colors=magenta, labelsize=13, length=6, width=1.2)
    for s in axr.spines.values():
        s.set_color(white)
        s.set_linewidth(1.2)

    ax.set_xlim(float(np.min(bpw)) - 0.1, float(np.max(bpw)) + 0.1)
    ax.set_ylim(*pad(float(np.min(kld)), float(np.max(kld)), 0.10))
    axr.set_ylim(*pad(float(np.min(ppl)), float(np.max(ppl)), 0.10))

    ax.text(
        0.5,
        0.90,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color=cyan,
        fontsize=22,
        fontweight="bold",
    )

    handles = [l1, l2]
    labels = [h.get_label() for h in handles]
    leg = ax.legend(handles, labels, loc="upper right", frameon=False, fontsize=14, handlelength=2.6)
    for text in leg.get_texts():
        text.set_color(white)

    order = np.argsort(bpw)
    bpw_s = bpw[order]
    gib_s = gib[order]

    def bpw_to_gib(x):
        return np.interp(x, bpw_s, gib_s)

    def gib_to_bpw(x):
        return np.interp(x, gib_s, bpw_s)

    secax = ax.secondary_xaxis("top", functions=(bpw_to_gib, gib_to_bpw))
    secax.set_xlabel("Model Size (GiB)", color=white, fontsize=16, labelpad=12)
    secax.tick_params(axis="x", colors=white, labelsize=13, length=6, width=1.2)

    ticks, labels = _top_axis_ticks_and_labels(gib_s)
    if len(ticks) >= 2:
        secax.set_xticks(ticks)
        secax.set_xticklabels(labels)

    if add_checks:
        for x, y in zip(bpw, kld):
            ax.text(
                x - 0.06,
                y + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02,
                "✓",
                color=white,
                fontsize=32,
                fontweight="bold",
                ha="center",
                va="center",
            )

    plt.tight_layout()
    os.makedirs(os.path.dirname(outfile) or ".", exist_ok=True)
    fig.savefig(outfile, facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def generate_iceblink_svg(csv_path: str, out_svg: str, title: str) -> str:
    bpw, kld, ppl, gib, _ = load_series(csv_path, drop_bf16=True)
    if len(bpw) < 2:
        raise ValueError("Need at least 2 non-bf16 rows to draw graph")
    make_plot(bpw, kld, ppl, gib, title=title, outfile=out_svg, add_checks=True)
    return out_svg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Input CSV path")
    p.add_argument(
        "--out",
        default="iceblink_style.svg",
        help="Output image path (e.g., .svg or .png)",
    )
    p.add_argument("--title", default="Iceblink-v2-exl3", help="Title text inside plot")
    p.add_argument("--no-checks", action="store_true", help="Disable checkmark annotations")
    p.add_argument("--keep-bf16", action="store_true", help="Keep bf16 row (otherwise dropped)")
    args = p.parse_args()

    bpw, kld, ppl, gib, _ = load_series(args.csv, drop_bf16=not args.keep_bf16)
    make_plot(bpw, kld, ppl, gib, title=args.title, outfile=args.out, add_checks=not args.no_checks)


if __name__ == "__main__":
    main()
