#!/usr/bin/env python3
"""Organize N-sigma plots into pass-threshold folders 4, 3, and 2.

The wrapper reads target_nsigma_pixel_results.csv from the output root. It
moves existing PNGs into the ``4`` folder, then writes threshold-specific
pass/fail donut plots for minimum pixel counts >=4, >=3, and >=2. Other plots
are copied to the >=3 and >=2 folders because they do not depend on the pass
cut. No spectra are reloaded or N_sigma values recalculated.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ROOT = Path(
    "/work/11161/kanyuni/ls6/quassiQ_project/"
    "pipeline_output/post_processing/nsigma_pixel"
)
LEVELS = ["[1, 2.5)", "[2.5, 3.5)", "[3.5, 5]"]
THRESHOLDS = (4, 3, 2)


def category_filename(category: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", category).strip("_").lower()


def make_summary(targets: pd.DataFrame, min_pixels: int) -> pd.DataFrame:
    frame = targets.copy()
    frame["max_pixels_nsigma_gt_3"] = pd.to_numeric(
        frame["max_pixels_nsigma_gt_3"], errors="coerce"
    ).fillna(0)
    frame["passes_new_threshold"] = frame["max_pixels_nsigma_gt_3"] >= min_pixels
    summary = (
        frame[frame["average_level"].isin(LEVELS)]
        .groupby(["average_level", "passes_new_threshold"])
        .size()
        .rename("target_count")
        .reset_index()
    )
    full = pd.MultiIndex.from_product(
        [LEVELS, [False, True]],
        names=["average_level", "passes_new_threshold"],
    )
    return (
        summary.set_index(["average_level", "passes_new_threshold"])
        .reindex(full, fill_value=0)
        .reset_index()
    )


def make_donut(summary: pd.DataFrame, path: Path, min_pixels: int, title: str) -> None:
    pivot = summary.pivot(
        index="average_level", columns="passes_new_threshold", values="target_count"
    ).reindex(LEVELS, fill_value=0)
    pivot = pivot.reindex(columns=[False, True], fill_value=0)
    level_colors = ["#f28e8e", "#ffbf78", "#79dd83"]
    inner_counts = []
    outer_counts = []
    outer_colors = []
    outer_labels = []
    for level, color in zip(LEVELS, level_colors):
        fail_count = int(pivot.loc[level, False])
        pass_count = int(pivot.loc[level, True])
        inner_counts.append(fail_count + pass_count)
        outer_counts.extend([fail_count, pass_count])
        outer_colors.extend([color + "88", color])
        outer_labels.extend([
            f"{level} — Fail (<{min_pixels} pixels)",
            f"{level} — Pass (≥{min_pixels} pixels)",
        ])

    fig, ax = plt.subplots(figsize=(10, 7))
    total = sum(inner_counts)
    if total:
        wedges, _, _ = ax.pie(
            outer_counts,
            radius=1.0,
            colors=outer_colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.30, "edgecolor": "white", "linewidth": 1.2},
            autopct=lambda pct: f"{int(round(pct * total / 100))}" if pct >= 4 else "",
            pctdistance=0.85,
            textprops={"fontsize": 9},
        )
        ax.pie(
            inner_counts,
            radius=0.69,
            labels=[f"{level}\nn={count}" for level, count in zip(LEVELS, inner_counts)],
            colors=level_colors,
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.30, "edgecolor": "white", "linewidth": 1.2},
            labeldistance=0.76,
            textprops={"fontsize": 8, "ha": "center"},
        )
        ax.legend(
            wedges,
            outer_labels,
            title="Outer ring: threshold result by score level",
            loc="center left",
            bbox_to_anchor=(1.0, 0.5),
            fontsize="small",
        )
        ax.text(0, 0, f"All targets\n{total}", ha="center", va="center", fontsize=11)
    else:
        ax.text(0, 0, "No scored targets\nto plot", ha="center", va="center", fontsize=12)
    ax.set_title(f"{title}\nPass if at least {min_pixels} pixels have N_sigma > 3")
    ax.set_aspect("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def move_root_pngs_to_four(root: Path, four_dir: Path) -> None:
    four_dir.mkdir(parents=True, exist_ok=True)
    for source in root.glob("*.png"):
        destination = four_dir / source.name
        if source.resolve() != destination.resolve():
            shutil.move(str(source), str(destination))


def copy_threshold_independent_plots(four_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for source in four_dir.glob("*.png"):
        if source.name.startswith("pass_counts_by_average_level"):
            continue
        shutil.copy2(source, destination / source.name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Folder containing target_nsigma_pixel_results.csv and current PNGs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root
    csv_path = root / "target_nsigma_pixel_results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Target-results CSV not found: {csv_path}")
    targets = pd.read_csv(csv_path)
    required = {"category", "average_level", "max_pixels_nsigma_gt_3"}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"Target-results CSV is missing columns: {sorted(missing)}")

    dirs = {threshold: root / str(threshold) for threshold in THRESHOLDS}
    move_root_pngs_to_four(root, dirs[4])
    for threshold in (3, 2):
        copy_threshold_independent_plots(dirs[4], dirs[threshold])

    for threshold in THRESHOLDS:
        for old_plot in dirs[threshold].glob("pass_counts_by_average_level*.png"):
            old_plot.unlink()
        summary = make_summary(targets, threshold)
        make_donut(
            summary,
            dirs[threshold] / "pass_counts_by_average_level.png",
            threshold,
            "All categories: N_sigma pass/fail by Average-score level",
        )
        for category in sorted(targets["category"].dropna().unique()):
            subset = targets[targets["category"] == category]
            category_summary = make_summary(subset, threshold)
            suffix = category_filename(str(category))
            make_donut(
                category_summary,
                dirs[threshold] / f"pass_counts_by_average_level_{suffix}.png",
                threshold,
                f"{category}: N_sigma pass/fail by Average-score level",
            )
        print(f">={threshold} pixels: {dirs[threshold]}")


if __name__ == "__main__":
    main()
