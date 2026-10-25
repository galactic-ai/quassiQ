#!/usr/bin/env python3
"""Count significant N-sigma pixels for selected targets and score levels.

For each TARGETID row on each category sheet in CLQ_visual.xlsx, this script:
  1. Loads the target's reconstructed epochs with the N_sigma pipeline.
  2. Computes N_sigma for every pair of epochs.
  3. Counts finite pixels with N_sigma > 3 within selected emission-line windows.
  4. Evaluates minimum significant-pixel counts of 4, 3, and 2.
  5. Joins the target to its Average score from CLQ_visual.xlsx and assigns a level.

The imported pipeline's calculate_n_sigma() controls the weights and N_sigma
definition. This script does not modify or unnormalize those weights.
"""

from __future__ import annotations

import argparse
import importlib.util
from itertools import combinations
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_NSIGMA = Path(
    "/work/11161/kanyuni/ls6/quassiQ_project/"
    "quassiQ/src/pipeline/N_sigma.py"
)
DEFAULT_VISUAL_XLSX = Path(
    "/work/11161/kanyuni/ls6/quassiQ_project/"
    "pipeline_output/overlap/CLQ_visual.xlsx"
)
DEFAULT_OUTPUT = Path(
    "/work/11161/kanyuni/ls6/quassiQ_project/"
    "pipeline_output/post_processing/pixel"
)

C_KMS = 299792.458
LINE_HALF_WIDTH_KMS = 5000.0

EMISSION_LINES = {
    "lya":  {"label": "Ly-alpha", "wavelength": 1215.67},
    "civ":  {"label": "C IV",     "wavelength": 1549.48},
    "mgii": {"label": "Mg II",    "wavelength": 2798.75},
}

NSIGMA_THRESHOLD = 3.0
THRESHOLDS = (4, 3, 2)
LEVELS = [
    ("[1, 2.5)", 1.0, 2.5, False),
    ("[2.5, 3.5)", 2.5, 3.5, False),
    ("[3.5, 5]", 3.5, 5.0, True),
]
LEVEL_LABELS = [level[0] for level in LEVELS]


def normalize_target_id(value) -> str | None:
    """Normalize numeric/string TARGETIDs without converting them to floats."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        # This fallback supports integer-valued numeric cells. For full
        # 17-digit accuracy, TARGETID cells should be stored as text in Excel.
        return str(int(round(float(value))))
    value = str(value).strip()
    if not value:
        return None
    # Excel sometimes exports integer IDs as strings ending in .0.
    if re.fullmatch(r"\d+\.0", value):
        value = value[:-2]
    elif re.fullmatch(r"\d+(?:\.\d+)?[eE][+-]?\d+", value):
        try:
            value = str(int(round(float(value))))
        except (ValueError, OverflowError):
            return None
    return value


def load_pipeline(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"N_sigma script not found: {path}")
    spec = importlib.util.spec_from_file_location("nsigma_pipeline", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import N_sigma script: {path}")
    module = importlib.util.module_from_spec(spec)
    # Register before execution so decorators/type annotations resolve normally.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    required = ("load_candidate_epochs", "calculate_n_sigma")
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            f"{path} is missing required function(s): {', '.join(missing)}"
        )
    return module


def category_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
    return key[:-4] if key.endswith("only") else key


def read_workbook_targets(path: Path, requested_categories=None) -> list[dict]:
    """Read TARGETID and Average directly from each category worksheet."""
    if not path.is_file():
        raise FileNotFoundError(f"Visual-inspection workbook not found: {path}")
    workbook = pd.ExcelFile(path)
    requested = {category_key(name) for name in requested_categories or []}
    rows = []
    found_category_sheets = []
    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
        by_name = {str(column).strip().casefold(): column for column in frame.columns}
        id_column = next(
            (by_name[name] for name in ("targetid", "target_id", "id") if name in by_name),
            None,
        )
        average_column = next(
            (by_name[name] for name in ("average", "avg") if name in by_name),
            None,
        )
        if id_column is None or average_column is None:
            continue  # Skip the workbook's summary/result sheet.
        if requested and category_key(sheet_name) not in requested:
            continue
        found_category_sheets.append(sheet_name)
        for raw_id, raw_score in zip(frame[id_column], frame[average_column]):
            target_id = normalize_target_id(raw_id)
            if target_id is None:
                continue
            score = pd.to_numeric(raw_score, errors="coerce")
            rows.append({
                "category": sheet_name,
                "workbook_targetid": target_id,
                "Average": float(score) if pd.notna(score) else np.nan,
            })
    if requested:
        missing = requested - {category_key(name) for name in found_category_sheets}
        if missing:
            raise ValueError(
                "Requested category sheet(s) were not found: " + ", ".join(sorted(missing))
            )
    if not found_category_sheets:
        raise ValueError(
            "No selected workbook sheet has both TARGETID and Average columns. "
            f"Sheets found: {workbook.sheet_names}"
        )
    # Duplicate rounded IDs within one sheet collapse to one target row; average
    # their visible scores, preserving a blank score if all rows are blank.
    if not rows:
        raise ValueError("The selected worksheets contain no usable TARGETIDs.")
    records = pd.DataFrame(rows)
    records = (records.groupby(["category", "workbook_targetid"], as_index=False)
               .agg(Average=("Average", "mean"), workbook_rows=("Average", "size")))
    return records.to_dict("records")


def score_level(score: float) -> str:
    if not np.isfinite(score):
        return "missing"
    for label, lower, upper, include_upper in LEVELS:
        if lower <= score and (score <= upper if include_upper else score < upper):
            return label
    return "outside ranges"

def emission_window_mask(wave):
    wave = np.asarray(wave, dtype=float)
    selected = np.zeros(wave.shape, dtype=bool)

    for line in EMISSION_LINES.values():
        center = line["wavelength"]
        half_width = center * LINE_HALF_WIDTH_KMS / C_KMS

        selected |= (
            (wave >= center - half_width)
            & (wave <= center + half_width)
        )

    return selected


def count_significant_pixels(calculation: dict) -> tuple[int, int]:
    values = np.asarray(calculation["n_sigma"], dtype=float)

    selected = emission_window_mask(calculation["wave"])
    finite = np.isfinite(values) & selected

    return (
        int(np.count_nonzero(finite & (values > NSIGMA_THRESHOLD))),
        int(finite.sum()),
    )


def analyze_target(pipeline, target_id: str, category: str) -> tuple[dict, list[dict]]:
    epochs = pipeline.load_candidate_epochs(target_id)
    pair_results = []
    max_nsigma = np.nan
    max_nsigma_wave = np.nan
    for first, second in combinations(epochs, 2):
        calculation = pipeline.calculate_n_sigma(first, second)
        pixels_over, finite_pixels = count_significant_pixels(calculation)
        n_sigma = np.asarray(calculation["n_sigma"], dtype=float)
        finite = (np.isfinite(n_sigma)& emission_window_mask(calculation["wave"]))
        if np.any(finite):
            peak_index = np.flatnonzero(finite)[np.argmax(n_sigma[finite])]
            pair_peak = float(n_sigma[peak_index])
            if not np.isfinite(max_nsigma) or pair_peak > max_nsigma:
                max_nsigma = pair_peak
                max_nsigma_wave = float(calculation["wave"][peak_index])
        pair_results.append({
            "category": category,
            "TARGETID": target_id,
            "first_date": first.get("date", ""),
            "second_date": second.get("date", ""),
            "pixels_nsigma_gt_3": pixels_over,
            "finite_pixels_compared": finite_pixels,
            "max_nsigma_in_pair":(float(np.max(n_sigma[finite])) if np.any(finite) else np.nan),
        })

    max_pixels = max(
        (row["pixels_nsigma_gt_3"] for row in pair_results),
        default=0,
    )
    target_result = {
        "category": category,
        "TARGETID": target_id,
        "number_of_epochs": len(epochs),
        "number_of_pairs": len(pair_results),
        "max_pixels_nsigma_gt_3": max_pixels,
        "max_nsigma_all_pairs": max_nsigma,
        "wavelength_at_max_nsigma": max_nsigma_wave,
        "best_pair_first_date": "",
        "best_pair_second_date": "",
    }
    if pair_results:
        best_pair = max(pair_results, key=lambda row: row["pixels_nsigma_gt_3"])
        target_result["best_pair_first_date"] = best_pair["first_date"]
        target_result["best_pair_second_date"] = best_pair["second_date"]
    return target_result, pair_results


def make_summary(target_results: pd.DataFrame, min_pixels: int) -> pd.DataFrame:
    frame = target_results.copy()

    frame["max_pixels_nsigma_gt_3"] = pd.to_numeric(
        frame["max_pixels_nsigma_gt_3"], errors="coerce"
    ).fillna(0)

    # Recalculate pass/fail for this particular pixel requirement.
    frame["passes_new_threshold"] = (
        frame["max_pixels_nsigma_gt_3"] >= min_pixels
    )

    summary = (
        frame[frame["average_level"].isin(LEVEL_LABELS)]
        .groupby(["average_level", "passes_new_threshold"])
        .size()
        .rename("target_count")
        .reset_index()
    )

    # Include all three levels, each with a fail and a pass row.
    full = pd.MultiIndex.from_product(
        [LEVEL_LABELS, [False, True]],
        names=["average_level", "passes_new_threshold"],
    )

    return (
        summary.set_index(["average_level", "passes_new_threshold"])
        .reindex(full, fill_value=0)
        .reset_index()
    )




def make_category_correlation_plot(
    target_results: pd.DataFrame,
    category: str | None,
    y_column: str,
    y_label: str,
    title: str,
    path: Path,
    *,
    threshold_line: float | None = None,
) -> None:
    """Scatter Average score against a target-level N_sigma summary."""
    if category is None:
        data = target_results.copy()
    else:
        data = target_results[target_results["category"] == category].copy()
    data["Average"] = pd.to_numeric(data["Average"], errors="coerce")
    data[y_column] = pd.to_numeric(data[y_column], errors="coerce")
    data = data.dropna(subset=["Average", y_column])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    if not data.empty:
        categories = sorted(data["category"].dropna().unique())
        colors = plt.get_cmap("tab10")
        for index, plot_category in enumerate(categories):
            group = data[data["category"] == plot_category]
            ax.scatter(
                group["Average"], group[y_column],
                s=34, alpha=0.75, color=colors(index % 10),
                label=f"{plot_category} (n={len(group)})",
            )

    if len(data) >= 2 and data["Average"].nunique() > 1 and data[y_column].nunique() > 1:
        pearson_r = float(data["Average"].corr(data[y_column], method="pearson"))
        slope, intercept = np.polyfit(data["Average"], data[y_column], 1)
        x_line = np.linspace(data["Average"].min(), data["Average"].max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.3,
                label=f"Linear fit (Pearson r={pearson_r:.2f}, n={len(data)})")
    else:
        ax.text(0.03, 0.97, f"Not enough variation to calculate correlation (n={len(data)})",
                transform=ax.transAxes, va="top")

    if threshold_line is not None:
        ax.axhline(threshold_line, color="tab:red", linestyle="--", linewidth=1,
                   label=f"N_sigma = {threshold_line:g}")
    ax.set_xlabel("Average visual-inspection score")
    ax.set_ylabel(y_label)
    ax.set_title(f"{category}: {title}" if category else f"All categories: {title}")
    ax.grid(alpha=0.25)
    ax.legend(fontsize="small", loc="best")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def category_filename(category: str) -> str:
    """Make a readable, filesystem-safe category suffix."""
    return re.sub(r"[^A-Za-z0-9_-]+", "_", category).strip("_").lower()




def make_donut(summary: pd.DataFrame, path: Path, min_pixels: int, title: str) -> None:
    pivot = summary.pivot(
        index="average_level", columns="passes_new_threshold", values="target_count"
    ).reindex(LEVEL_LABELS, fill_value=0)
    pivot = pivot.reindex(columns=[False, True], fill_value=0)
    level_colors = ["#f28e8e", "#ffbf78", "#79dd83"]
    inner_counts = []
    outer_counts = []
    outer_colors = []
    outer_labels = []
    for level, color in zip(LEVEL_LABELS, level_colors):
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
            labels=[f"{level}\nn={count}" for level, count in zip(LEVEL_LABELS, inner_counts)],
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nsigma-script", type=Path, default=DEFAULT_NSIGMA)
    parser.add_argument("--visual-xlsx", type=Path, default=DEFAULT_VISUAL_XLSX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Optional category worksheet names; default uses all TARGETID worksheets.")               
    return parser.parse_args()

def main():
    args = parse_args()
    pipeline = load_pipeline(args.nsigma_script)
    workbook_targets = read_workbook_targets(args.visual_xlsx, args.categories)
    dirs = {threshold: args.output_dir / str(threshold) for threshold in THRESHOLDS}
    target_results = []
    pair_results = []
    for row in workbook_targets:
        category = row["category"]
        workbook_targetid = row["workbook_targetid"]
        target_id = workbook_targetid
        result, pairs = analyze_target(pipeline, target_id, category)
        average = row["Average"]
        result["workbook_TARGETID"] = workbook_targetid
        result["workbook_rows_for_id"] = row["workbook_rows"]
        result["Average"] = average
        result["average_level"] = score_level(average)
        target_results.append(result)
        for pair in pairs:
            pair["workbook_TARGETID"] = workbook_targetid
            pair["Average"] = average
            pair["average_level"] = result["average_level"]
            pair_results.append(pair)
        print(
            f"{category} TARGETID={target_id}: "
            f"epochs={result['number_of_epochs']} "
            f"pairs={result['number_of_pairs']} "
            f"max_pixels>{NSIGMA_THRESHOLD:g}="
            f"{result['max_pixels_nsigma_gt_3']} "
            f"level={result['average_level']}"
        )

    targets = pd.DataFrame(target_results)
    pairs = pd.DataFrame(pair_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    targets.to_csv(args.output_dir / "target_nsigma_pixel_results.csv", index=False)
    pairs.to_csv(args.output_dir / "epoch_pair_nsigma_pixel_results.csv", index=False)

    make_category_correlation_plot(
        targets,
        None,
        "max_pixels_nsigma_gt_3",
        "Maximum pixels with N_sigma > 3 in any epoch pair",
        "Visual score vs significant-pixel count",
        args.output_dir / "emission_score_vs_nsigma_pixel_count.png",
    )
    make_category_correlation_plot(
        targets,
        None,
        "max_nsigma_all_pairs",
        "Highest N_sigma pixel across all epoch pairs",
        "Visual score vs highest N_sigma",
        args.output_dir / "emission_score_vs_max_nsigma.png",
        threshold_line=NSIGMA_THRESHOLD,
    )
    # The CSV outputs above also contain combined results for all categories.
    for category in sorted(targets["category"].dropna().unique()):
        category_targets = targets[targets["category"] == category]
        suffix = category_filename(category)
        make_category_correlation_plot(
            category_targets,
            category,
            "max_pixels_nsigma_gt_3",
            "Maximum pixels with N_sigma > 3 in any epoch pair",
            "Visual score vs significant-pixel count",
            args.output_dir / f"score_vs_nsigma_pixel_count_{suffix}.png",
        )
        make_category_correlation_plot(
            category_targets,
            category,
            "max_nsigma_all_pairs",
            "Highest N_sigma pixel across all epoch pairs",
            "Visual score vs highest N_sigma",
            args.output_dir / f"score_vs_max_nsigma_{suffix}.png",
            threshold_line=NSIGMA_THRESHOLD,
        )




    for threshold in THRESHOLDS:
        output_dir = dirs[threshold]
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = make_summary(targets, threshold)
        summary.to_csv(
            output_dir / "pass_counts_by_average_level.csv",
            index=False,
        )

        make_donut(
            summary,
            output_dir / "pass_counts_by_average_level.png",
            threshold,
            "All categories: N_sigma pass/fail by Average-score level",
        )

        category_summaries = []

        for category in sorted(targets["category"].dropna().unique()):
            subset = targets[targets["category"] == category]
            category_summary = make_summary(subset, threshold)
            suffix = category_filename(str(category))

            category_summary.to_csv(
                output_dir / f"pass_counts_by_average_level_{suffix}.csv",
                index=False,
            )

            # Add the category name before combining category tables.
            category_summaries.append(
                category_summary.assign(category=category)
            )

            make_donut(
                category_summary,
                output_dir / f"pass_counts_by_average_level_{suffix}.png",
                threshold,
                f"{category}: N_sigma pass/fail by Average-score level",
            )

        pd.concat(category_summaries, ignore_index=True).to_csv(
            output_dir / "pass_counts_by_category_and_level.csv",
            index=False,
        )

        print(f"\nPass counts for >= {threshold} significant pixels:")
        print(summary.to_string(index=False))
        print(f"Outputs written to: {output_dir}")




if __name__ == "__main__":
    main()
