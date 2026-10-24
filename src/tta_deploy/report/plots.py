from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tta_deploy.data.imagenet_c import CORRUPTIONS
from tta_deploy.eval.metrics import (
    rolling_accuracy,
    rolling_class_histogram_entropy,
    rolling_prediction_entropy,
    window_end_frames,
)
from tta_deploy.report.load import Run
from tta_deploy.report.tables import average, order

SERIES_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
OVER_TIME_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#6250d6",
    "#e87ba4",
    "#eda100",
    "#e34948",
)
SOURCE_COLOR = "#52514e"
# The TENT grid sweeps S at this B (plan, Experiments).
STRIDE_SWEEP_B = 16


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_knobs(runs: dict[str, Run], out: Path) -> None:
    source = average(runs["source"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    tent = [r.metrics for n, r in runs.items() if n != "source"]
    stride_sweep_batches = sorted({r["B"] for r in tent if r["S"] > 1})
    panels = [
        (
            axes[0],
            "B (S = 1)",
            "B",
            [("S = 1", [r for r in tent if r["S"] == 1])],
        ),
        (
            axes[1],
            "S",
            "S",
            [
                (f"B = {B}", [r for r in tent if r["B"] == B])
                for B in stride_sweep_batches
            ],
        ),
    ]
    for ax, xlabel, key, groups in panels:
        assert len(groups) <= len(SERIES_COLORS), (
            f"more series than colors: {len(groups)}"
        )
        for color, (label, group) in zip(SERIES_COLORS, groups, strict=False):
            group = sorted(group, key=lambda r: r[key])
            if not group:
                continue
            xs, ys = (
                [r[key] for r in group],
                [r["accuracy"]["average"] for r in group],
            )
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2,
                marker="o",
                markersize=6,
                label=label,
            )
            ax.annotate(
                label,
                (xs[-1], ys[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color="#0b0b0b",
            )
        ax.axhline(source, color=SOURCE_COLOR, linewidth=1.2, linestyle="--")
        ax.annotate(
            f"source {source:.1f}",
            (0, source),
            xycoords=("axes fraction", "data"),
            xytext=(2, 4),
            textcoords="offset points",
            fontsize=9,
            color=SOURCE_COLOR,
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        style(ax)
        if len(groups) > 1:
            ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("15-corruption average top-1 (%)")
    fig.suptitle("Reference accuracy vs adaptation batch B and stride S", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cost_vs_knobs(
    max_fps: dict[str, list[tuple[int, int, float]]],
    p99_at_30_fps: dict[str, list[tuple[int, int, float]]],
    source_max_fps: dict[str, float],
    out: Path,
) -> None:
    """Per method kind: (B, S, value) of each TENT config. Source max
    fps per kind is the ceiling."""
    kinds = sorted(max_fps.keys() | p99_at_30_fps.keys()) + sorted(source_max_fps)
    assert len(kinds) <= len(SERIES_COLORS), f"more kinds than colors: {kinds}"
    colors = dict(zip(kinds, SERIES_COLORS, strict=False))
    rows = [
        (max_fps, "max sustainable fps"),
        (p99_at_30_fps, "p99 latency at 30 fps (ms)"),
    ]
    columns = [
        ("B (S = 1)", 1, None),
        (f"S (B = {STRIDE_SWEEP_B})", None, STRIDE_SWEEP_B),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex="col")
    for row, (points, ylabel) in enumerate(rows):
        for col, (xlabel, S_fixed, B_fixed) in enumerate(columns):
            ax = axes[row, col]
            for kind, series in sorted(points.items()):
                xy = sorted(
                    (B if S_fixed else S, value)
                    for B, S, value in series
                    if S == S_fixed or B == B_fixed
                )
                if xy:
                    xs, ys = zip(*xy, strict=True)
                    ax.plot(
                        xs,
                        ys,
                        color=colors[kind],
                        linewidth=2,
                        marker="o",
                        markersize=6,
                        label=kind,
                    )
            if row == 0:
                for kind, fps in sorted(source_max_fps.items()):
                    ax.axhline(
                        fps,
                        color=colors[kind],
                        linewidth=1.2,
                        linestyle="--",
                        label=kind,
                    )
                ax.set_yscale("log")
            ax.set_xscale("log", base=2)
            style(ax)
            if row == len(rows) - 1:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(ylabel)
    # One legend for every panel: a kind may lack runs in some.
    handles = {
        label: handle
        for ax in axes.flat
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=True)
    }
    fig.legend(
        handles.values(),
        handles.keys(),
        frameon=False,
        fontsize=8,
        ncol=len(handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.suptitle("Cost vs adaptation batch B and stride S (RTX 3080)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_over_time(runs: dict[str, Run], out: Path) -> None:
    groups = [
        (
            "S = 1, varying B",
            [n for n in order(runs) if n == "source" or runs[n].metrics["S"] == 1],
        ),
        (
            f"B = {STRIDE_SWEEP_B}, varying S",
            [
                n
                for n in order(runs)
                if n == "source" or runs[n].metrics["B"] == STRIDE_SWEEP_B
            ],
        ),
    ]
    # Fixed color per config, so a config keeps its color in every
    # panel.
    tent_names = [
        n
        for n in order(runs)
        if n != "source" and any(n in names for _, names in groups)
    ]
    assert len(tent_names) <= len(OVER_TIME_COLORS), (
        f"more configs than colors: {tent_names}"
    )
    colors = dict(zip(tent_names, OVER_TIME_COLORS, strict=False)) | {
        "source": SOURCE_COLOR
    }
    rows = [
        (
            "rolling top-1 (%)",
            lambda f: rolling_accuracy(f["predictions"], f["labels"]),
        ),
        (
            "rolling prediction entropy (nats)",
            lambda f: rolling_prediction_entropy(f["entropies"]),
        ),
        (
            "rolling predicted-class histogram entropy (nats)",
            lambda f: rolling_class_histogram_entropy(f["predictions"]),
        ),
    ]
    frames = {n: runs[n].frames() for n in {"source", *tent_names}}
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 9), sharex=True)
    num_frames = len(frames["source"]["labels"])
    x = window_end_frames(num_frames)
    for col, (title, names) in enumerate(groups):
        for row, (ylabel, metric) in enumerate(rows):
            ax = axes[row, col]
            for name in names:
                ax.plot(
                    x,
                    metric(frames[name]),
                    color=colors[name],
                    linewidth=1.2 if name == "source" else 2,
                    label=name,
                )
            for boundary in range(
                num_frames // len(CORRUPTIONS),
                num_frames,
                num_frames // len(CORRUPTIONS),
            ):
                ax.axvline(boundary, color="#e6e5e0", linewidth=0.8)
            style(ax)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
        axes[0, col].legend(
            frameon=False,
            fontsize=8,
            ncol=len(names),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            title=title,
            title_fontsize=10,
        )
        axes[-1, col].set_xlabel("frame (corruption boundaries in gray)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_accuracy_vs_fps(
    points: dict[str, list[tuple[float, float, str]]],
    source: float,
    out: Path,
) -> None:
    """Per method kind: (max fps, accuracy, label) of each config."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    assert len(points) <= len(SERIES_COLORS), f"more kinds than colors: {points}"
    for color, (kind, series) in zip(SERIES_COLORS, points.items(), strict=False):
        xs, ys, labels = zip(*sorted(series), strict=True)
        ax.plot(xs, ys, color=color, linewidth=0, marker="o", markersize=6, label=kind)
        for x, y, label in series:
            ax.annotate(
                label, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=7
            )
    ax.axhline(source, color=SOURCE_COLOR, linewidth=1.2, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("max sustainable fps (RTX 3080, ADR 0002)")
    ax.set_ylabel("15-corruption average top-1 (%)")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
