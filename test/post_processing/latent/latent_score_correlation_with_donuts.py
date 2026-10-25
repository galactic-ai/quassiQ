#!/usr/bin/env python3
"""Plot visual-inspection Average score against mean AND maximum latent variance.

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
    # Require individual dimensions so both the mean and maximum can be computed.
    
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
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    # Store the mean across latent dimensions as the one plotted value. Keep
    # the individual values in the matched-data CSV for auditability.
    frame["latent_variance"] = frame[variance_columns].mean(axis=1)
    frame["variance_dimensions_used"] = frame[variance_columns].notna().sum(axis=1)
    frame = frame.dropna(subset=["TARGETID", "latent_variance"])
    # Collapse repeated target rows by averaging each dimension first.
    grouped = frame.groupby("TARGETID", as_index=False)[variance_columns].mean()
    grouped["latent_variance"] = grouped[variance_columns].mean(axis=1)
    grouped["max_latent_variance"] = grouped[variance_columns].max(axis=1, skipna=False)
    grouped["variance_dimensions_used"] = grouped[variance_columns].notna().sum(axis=1)
    return grouped.set_index("TARGETID").to_dict(orient="index")


def lookup_variance(target_id: str, lookup) -> tuple[float, str]:
    if target_id in lookup:
        return lookup[target_id], "exact"
    return {
        "latent_variance": np.nan,
        "max_latent_variance": np.nan,
        "variance_dimensions_used": 0,
    }, "not_found"


def make_plot(
    data: pd.DataFrame,
    output_path: Path,
    category: str | None = None,
    value_column: str = "latent_variance",
) -> None:
    if value_column not in {"latent_variance", "max_latent_variance"}:
        raise ValueError(f"Unknown variance metric: {value_column}")
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
    metric = (
        "Maximum latent variance"
        if value_column == "max_latent_variance"
        else "Mean latent variance"
    )

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


def make_donut_plot(
    data: pd.DataFrame,
    output_path: Path,
    category: str,
    value_column: str,
    cut: float,
) -> list[dict]:
    """Nested score-level and pass/fail rings matching the pixel script."""
    score = pd.to_numeric(data["Average"], errors="coerce")
    variance = pd.to_numeric(data[value_column], errors="coerce")
    finite = np.isfinite(score) & np.isfinite(variance)
    in_range = score.between(1, 5)
    usable = data.loc[finite & in_range].copy()
    usable["Average"] = score.loc[usable.index]
    usable[value_column] = variance.loc[usable.index]
    if (usable[value_column] < 0).any():
        raise ValueError("Latent variance must be nonnegative.")

    rows = []
    levels = [
        ("[1, 2.5)", 1.0, 2.5, False),
        ("[2.5, 3.5)", 2.5, 3.5, False),
        ("[3.5, 5]", 3.5, 5.0, True),
    ]
    for label, lower, upper, include_upper in levels:
        mask = (usable["Average"] >= lower) & (
            (usable["Average"] <= upper) if include_upper else (usable["Average"] < upper)
        )
        group = usable.loc[mask]
        passed = group[value_column] >= cut
        for passes in [False, True]:
            count = int((passed == passes).sum())
            rows.append({
                "category": category, "metric": value_column, "variance_cut": cut,
                "average_level": label, "passes_new_threshold": passes,
                "target_count": count, "level_total": len(group),
                "percent_within_score_level": 100 * count / len(group) if len(group) else np.nan,
                "percent_of_all_usable": 100 * count / len(usable) if len(usable) else np.nan,
                "n_input": len(data), "n_usable": len(usable),
                "n_omitted_nonfinite": int((~finite).sum()),
                "n_omitted_outside_score_range": int((finite & ~in_range).sum()),
            })
    summary = pd.DataFrame(rows)
    metric = "Maximum" if value_column == "max_latent_variance" else "Mean"
    title = f"{category}: {metric.lower()} latent variance pass/fail by Average-score level"
    omitted = len(data) - len(usable)
    if omitted:
        title += f"\n({omitted} missing/nonfinite or out-of-range targets omitted)"
    draw_nested_donut(summary, output_path, cut, title)
    summary.to_csv(output_path.with_suffix(".csv"), index=False)
    return rows


LEVEL_LABELS = ["[1, 2.5)", "[2.5, 3.5)", "[3.5, 5]"]


def draw_nested_donut(summary: pd.DataFrame, path: Path, cut: float, title: str) -> None:
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
            f"{level} — Fail (<{cut:g})",
            f"{level} — Pass (≥{cut:g})",
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
    ax.set_title(f"{title}\nPass if latent variance ≥ {cut:g}")
    ax.set_aspect("equal")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def category_filename(category: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", category).strip("_").lower()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-xlsx", type=Path, default=DEFAULT_VISUAL_XLSX)
    parser.add_argument("--variance-csv", type=Path, default=DEFAULT_VARIANCE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variance-cut", type=float, default=0.03,
        help="Keep variance >= this value in the donuts (default: 0.03). Applied separately to mean and max.",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Category worksheet names to include; default uses all TARGETID worksheets.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not np.isfinite(args.variance_cut) or args.variance_cut < 0:
        raise ValueError("--variance-cut must be finite and nonnegative; use 0.03, not -3.")
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
        ("latent_variance", "score_vs_mean_latent_variance"),
        ("max_latent_variance", "score_vs_max_latent_variance"),
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


    donut_dir = args.output_dir / "donuts" / f"cut_{args.variance_cut:g}"
    donut_rows = []
    donut_groups = list(results.groupby("category", sort=True))
    donut_groups += [("All categories", results)] + comparison_groups
    for group_name, group_data in donut_groups:
        for value_column, metric in [
            ("latent_variance", "mean"),
            ("max_latent_variance", "max"),
        ]:
            donut_path = donut_dir / (
                f"donut_{metric}_latent_variance_{category_filename(group_name)}.png"
            )
            donut_rows.extend(make_donut_plot(
                group_data, donut_path, str(group_name), value_column, args.variance_cut
            ))
            plot_paths.append(donut_path)
    pd.DataFrame(donut_rows).to_csv(donut_dir / "donut_summary.csv", index=False)
    print(f"Donut summary: {donut_dir / 'donut_summary.csv'}")

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


