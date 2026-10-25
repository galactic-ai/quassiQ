#!/usr/bin/env python3
"""Plot visual-inspection Average score against latent variance.

TARGETIDs are read directly from the category worksheets in CLQ_visual.xlsx.
By default, all worksheets with TARGETID and Average columns are included; use
--categories to choose a subset. Variance values come from
latent_variance_by_target.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/work/11161/kanyuni/ls6/quassiQ_project")
DEFAULT_VISUAL_XLSX = PROJECT_ROOT / "pipeline_output/overlap/CLQ_visual.xlsx"
DEFAULT_VARIANCE_CSV = PROJECT_ROOT / "pipeline_output/latent/latent_variance_by_target.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "pipeline_output/post_processing/latent"


def normalize_target_id(value) -> str | None:
    """Return an exact ID token; reject numeric long IDs that may be rounded.

    Excel cannot preserve all digits of long IDs stored as numbers.  Rejecting
    those cells is safer than converting them to a different TARGETID and
    silently producing a false non-match.
    """
    if pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        target_id = str(int(value))
        if len(target_id.lstrip("-")) > 15:
            raise ValueError(
                f"TARGETID {target_id!r} is stored as a numeric cell with more "
                "than 15 digits. Restore the exact ID and store the Excel cell "
                "as text before running this script."
            )
        return target_id
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        if not float(value).is_integer():
            raise ValueError(f"TARGETID must be an integer, got numeric value {value!r}.")
        target_id = str(int(value))
        if len(target_id.lstrip("-")) > 15:
            raise ValueError(
                f"TARGETID {target_id!r} is stored as a numeric cell with more "
                "than 15 digits. Restore the exact ID and store the Excel cell "
                "as text before running this script."
            )
        return target_id
    value = str(value).strip()
    if not value:
        return None
    if re.fullmatch(r"\d+\.0", value):
        digits = value[:-2]
        if len(digits) > 15:
            raise ValueError(
                f"TARGETID {value!r} looks like a rounded numeric value. "
                "Restore the exact ID and store it as text."
            )
        value = digits
    elif re.fullmatch(r"\d+(?:\.\d+)?[eE][+-]?\d+", value):
        raise ValueError(
            f"TARGETID {value!r} is written in scientific notation. "
            "Use the exact full ID as text; do not convert it through a float."
        )
    return value


def category_key(value: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "", str(value).casefold())
    return key[:-4] if key.endswith("only") else key


def read_workbook_targets(path: Path, requested_categories=None) -> list[dict]:
    """Read TARGETID and Average directly from the selected category sheets."""
    if not path.is_file():
        raise FileNotFoundError(f"Visual-score workbook not found: {path}")
    workbook = pd.ExcelFile(path)
    requested = {category_key(name) for name in requested_categories or []}
    rows = []
    found_sheets = []
    for sheet in workbook.sheet_names:
        # Keep cell types so long numeric TARGETIDs can be rejected instead of
        # silently treated as exact strings after Excel has rounded them.
        frame = pd.read_excel(path, sheet_name=sheet, dtype=object)
        columns = {str(column).strip().casefold(): column for column in frame.columns}
        id_column = next(
            (columns[name] for name in ("targetid", "target_id", "id") if name in columns),
            None,
        )
        score_column = next(
            (columns[name] for name in ("average", "avg") if name in columns),
            None,
        )
        if id_column is None or score_column is None:
            continue
        if requested and category_key(sheet) not in requested:
            continue
        found_sheets.append(sheet)
        for raw_id, raw_score in zip(frame[id_column], frame[score_column]):
            target_id = normalize_target_id(raw_id)
            if target_id is None:
                continue
            score = pd.to_numeric(raw_score, errors="coerce")
            rows.append({
                "category": sheet,
                "TARGETID": target_id,
                "Average": float(score) if pd.notna(score) else np.nan,
            })
    if requested:
        missing = requested - {category_key(name) for name in found_sheets}
        if missing:
            raise ValueError("Requested category sheet(s) not found: " + ", ".join(sorted(missing)))
    if not found_sheets:
        raise ValueError("No selected workbook sheet contains TARGETID and Average columns.")
    records = pd.DataFrame(rows)
    conflicts = records.groupby(["category", "TARGETID"])["Average"].nunique(dropna=False)
    if (conflicts > 1).any():
        raise ValueError("Conflicting visual scores for repeated TARGETID within a workbook sheet. Fix the workbook first.")
    records = (records.groupby(["category", "TARGETID"], as_index=False)
               .agg(Average=("Average", "mean"), workbook_rows=("Average", "size")))
    return records.to_dict("records")


def find_variance_columns(frame: pd.DataFrame):
    columns = {str(column).strip().casefold(): column for column in frame.columns}
    id_column = next(
        (columns[name] for name in ("targetid", "target_id", "id") if name in columns),
        None,
    )
    if id_column is None:
        raise ValueError(f"Variance CSV has no TARGETID column. Columns: {list(frame.columns)}")
    # Accept either one aggregate field or one field per latent dimension.
    # For Latent1_var ... Latent10_var, read all dimensions and average them.
    
    dimension_columns = [
        column for column in frame.columns
        if re.fullmatch(
            r"latent\d+_var", str(column).strip(), flags=re.IGNORECASE
        )
    ]

    if len(dimension_columns) == 10:
        return id_column, dimension_columns

    raise ValueError(
        f"Expected 10 individual latent-variance columns; found "
        f"{len(dimension_columns)}."
    )



def read_variances(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Latent-variance CSV not found: {path}")
    frame = pd.read_csv(path, dtype=str)
    id_column, variance_columns = find_variance_columns(frame)
    frame = frame[[id_column, *variance_columns]].copy()
    frame["TARGETID"] = frame[id_column].map(normalize_target_id)
    for column in variance_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Store the mean across latent dimensions as the one plotted value. Keep
    # the individual values in the matched-data CSV for auditability.
    frame["latent_variance"] = frame[variance_columns].mean(axis=1)
    frame["variance_dimensions_used"] = frame[variance_columns].notna().sum(axis=1)
    frame = frame.dropna(subset=["TARGETID", "latent_variance"])
    # Collapse repeated target rows by averaging each dimension first.
    grouped = frame.groupby("TARGETID", as_index=False)[variance_columns].mean()
    grouped["latent_variance"] = grouped[variance_columns].mean(axis=1, skipna=False)
    grouped["median_latent_variance"] = grouped[variance_columns].median(axis=1, skipna=False)
    grouped["max_latent_variance"] = grouped[variance_columns].max(axis=1, skipna=False)
    grouped["variance_dimensions_used"] = grouped[variance_columns].notna().sum(axis=1)
    return grouped.set_index("TARGETID").to_dict(orient="index")


def lookup_variance(target_id: str, lookup) -> tuple[float, str]:
    if target_id in lookup:
        return lookup[target_id], "exact"
    return {
        "latent_variance": np.nan,
        "max_latent_variance": np.nan,
        "median_latent_variance": np.nan,
        "variance_dimensions_used": 0,
    }, "not_found"


def make_plot(
    data: pd.DataFrame,
    output_path: Path,
    category: str | None = None,
    value_column: str = "latent_variance",
    variance_cut: float | None = None,
) -> None:
    if value_column not in data.columns:
        raise ValueError(
            f"Missing {value_column}. Calculate it in read_variances() first."
        )

    usable = data.copy()
    if category is not None:
        usable = usable[usable["category"] == category]

    usable = usable[
        np.isfinite(usable["Average"])
        & np.isfinite(usable[value_column])
    ]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    categories = sorted(usable["category"].dropna().unique())
    colors = plt.get_cmap("tab10")

    for index, plot_category in enumerate(categories):
        group = usable[usable["category"] == plot_category]

        ax.scatter(
            group["Average"],
            group[value_column],
            s=34,
            alpha=0.72,
            color={"Non latent": "green",
                    "Latent involved": "orange",
                }.get(plot_category, colors(index % 10)),
            label=f"{plot_category} (n={len(group)})",
        )

    medians = score_medians(usable, value_column)
    if not medians.empty:
        ax.scatter(
            medians["Average"], medians["median_variance"],
            marker="D", s=65, color="#a63777", edgecolors="white",
            linewidths=0.7, zorder=5,
            label="Median across targets at each exact visual score",
        )

    if (
        len(usable) >= 2
        and usable["Average"].nunique() > 1
        and usable[value_column].nunique() > 1
    ):
        x = usable["Average"]
        y = usable[value_column]

        pearson = x.corr(y)
        spearman = x.rank().corr(y.rank())

        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)

        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="black",
            linewidth=1.4,
            label=(
                f"Pearson r={pearson:.2f}; "
                f"Spearman ρ={spearman:.2f}; n={len(usable)}"
            ),
        )
    else:
        ax.text(
            0.03,
            0.97,
            f"Correlation unavailable: insufficient data or constant values "
            f"(n={len(usable)})",
            transform=ax.transAxes,
            va="top",
        )
    if variance_cut is not None:
        ax.axhline(variance_cut, color="#666666", linestyle="--", linewidth=1.2,
                   label=f"Variance cut = {variance_cut:g}")
    metric = {
        "latent_variance": "Mean latent variance",
        "median_latent_variance": "Median latent variance",
        "max_latent_variance": "Maximum latent variance",
    }[value_column]

    group_label = category if category else "All categories"
    ax.set_title(f"{group_label}: visual score vs {metric.lower()}")
    ax.set_xlabel("Average visual-inspection score")
    ax.set_ylabel(f"{metric} across 10 dimensions")
    ax.grid(alpha=0.25)

    if categories:
        ax.legend(fontsize="small", loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def score_medians(data, value_column):
    """Summarize the plotted y metric across targets at each exact score."""
    valid = (np.isfinite(data["Average"]) & data["Average"].between(1, 5)
             & np.isfinite(data[value_column]))
    return (data.loc[valid].groupby("Average", as_index=False)
            .agg(n_targets=(value_column, "size"),
                 median_variance=(value_column, "median")))


def make_score_donut(data, path, levels, colors, group_name):
    """Inner ring: score bands; outer ring: updated categories within bands."""
    from matplotlib.patches import Patch

    valid_score = np.isfinite(data.Average) & data.Average.between(1, 5)
    usable = data.loc[valid_score].copy()
    categories = sorted(usable.category.unique())
    palette = plt.get_cmap("tab10")
    category_colors = {name: palette(i % 10) for i, name in enumerate(categories)}
    inner = [int((usable.score_level == level).sum()) for level in levels]
    total = sum(inner)
    outer, outer_colors, summary = [], [], []
    for level in levels:
        for category in categories:
            subset = usable.loc[(usable.score_level == level) & (usable.category == category)]
            count = len(subset)
            outer.append(count)
            outer_colors.append(category_colors[category])
            summary.append(dict(score_level=level, category=category, n_targets=count,
                                n_missing_variance=int((~np.isfinite(subset.latent_variance)).sum())))
    fig, ax = plt.subplots(figsize=(11, 8))
    if total:
        ax.pie(outer, radius=1., colors=outer_colors, startangle=90,
               counterclock=False,
               wedgeprops={"width": .30, "edgecolor": "white", "linewidth": 1.2},
               autopct=lambda pct: str(int(round(pct*total/100))) if pct >= 4 else "",
               pctdistance=.85, textprops={"fontsize": 9})
        ax.pie(inner, radius=.69, colors=colors, startangle=90, counterclock=False,
               labels=[f"{level}\nn={count}" if count/total >= .04 else ""
                       for level, count in zip(levels, inner)],
               wedgeprops={"width": .30, "edgecolor": "white", "linewidth": 1.2},
               labeldistance=.76, textprops={"fontsize": 8, "ha": "center"})
        ax.text(0, 0, f"{group_name}\n{total} scored targets", ha="center", va="center", fontsize=11)
        handles = [Patch(facecolor=color, label=f"Score {level}") for level, color in zip(levels, colors)]
        handles += [Patch(facecolor=category_colors[c], label=f"{c} (n={int((usable.category == c).sum())})")
                    for c in categories]
        ax.legend(handles=handles, title="Score bands / updated categories", loc="center left",
                  bbox_to_anchor=(1., .5), fontsize=9)
    else:
        ax.text(0, 0, "No scored targets", ha="center", va="center")
    ax.set_title(f"{group_name}: updated sample composition\nInner ring: visual score; outer ring: category within each score band")
    missing_variance = int((~np.isfinite(usable.latent_variance)).sum())
    fig.text(.02, .015, f"Invalid/missing scores omitted: {int((~valid_score).sum())}. "
             f"Scored targets without complete variance: {missing_variance} (included in composition).", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(summary, columns=["score_level", "category", "n_targets", "n_missing_variance"])


def category_filename(category: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", category).strip("_").lower()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-xlsx", type=Path, default=DEFAULT_VISUAL_XLSX)
    parser.add_argument("--variance-csv", type=Path, default=DEFAULT_VARIANCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Category worksheet names to include; default uses all TARGETID worksheets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    targets = read_workbook_targets(args.visual_xlsx, args.categories)
    variances = read_variances(args.variance_csv)
    rows = []
    for target in targets:
        target_id = target["TARGETID"]
        variance_info, variance_match = lookup_variance(target_id, variances)
        rows.append({
            **target,
            "score_match": "from_workbook_sheet",
            "variance_match": variance_match,
            **variance_info,
        })

    results = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "workbook_targets_score_latent_variance.csv"
    results.to_csv(csv_path, index=False)
    plot_paths = []

    for value_column, prefix in [
        ("latent_variance", "score_vs_latent_variance"),
        ("max_latent_variance", "score_vs_max_latent_variance"),
        ("median_latent_variance", "score_vs_median_latent_variance"),
    ]:
        combined_path = args.output_dir / f"{prefix}.png"
        make_plot(
            results,
            combined_path,
            value_column=value_column,
        )
        plot_paths.append(combined_path)

        for category in sorted(results["category"].dropna().unique()):
            plot_path = args.output_dir / (
                f"{prefix}_{category_filename(category)}.png"
            )
            make_plot(
                results,
                plot_path,
                category=category,
                value_column=value_column,
            )
            plot_paths.append(plot_path)
    # Identify all categories involving latent selection.
    latent_mask = results["category"].map(
        lambda name: (
            "latent" in category_key(name)
            or category_key(name) == "allthree"
        )
    )

    comparison_groups = [
        ("Latent involved", results.loc[latent_mask].copy()),
        ("Non latent", results.loc[~latent_mask].copy()),
    ]

    for group_name, group_data in comparison_groups:
        # Give this pooled group one label for the title and legend.
        group_data["category"] = group_name

        for value_column, metric in [
            ("latent_variance", "mean"),
            ("max_latent_variance", "max"),
            ("median_latent_variance", "median"),
        ]:
            plot_path = args.output_dir / (
                f"score_vs_{metric}_latent_variance_"
                f"{category_filename(group_name)}.png"
            )

            make_plot(
                group_data,
                plot_path,
                category=group_name,
                value_column=value_column,
            )

            plot_paths.append(plot_path)

    usable = results.dropna(subset=["Average", "latent_variance"])
    print(f"Workbook target rows: {len(results)}")
    print(f"Matched score and variance rows: {len(usable)}")
    if not results.empty:
        print("Score match status:")
        print(results["score_match"].value_counts(dropna=False).to_string())
        print("Variance match status:")
        print(results["variance_match"].value_counts(dropna=False).to_string())
    print(f"Matched-data CSV: {csv_path}")
    print("Category plots:")
    for plot_path in plot_paths:
        print(f"  {plot_path}")


if __name__ == "__main__":
    main()
