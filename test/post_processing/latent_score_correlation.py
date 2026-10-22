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
    exact = [columns[name] for name in ("latent_variance", "variance") if name in columns]
    if len(exact) == 1:
        variance_columns = exact
    else:
        variance_columns = [
            column for column in frame.columns
            if re.fullmatch(r"latent\d+_var", str(column).strip(), flags=re.IGNORECASE)
        ]
        if not variance_columns:
            candidates = [
                column for column in frame.columns
                if "variance" in str(column).casefold()
                and "normalized" not in str(column).casefold()
            ]
            if len(candidates) == 1:
                variance_columns = candidates
            else:
                raise ValueError(
                    "Could not identify a single variance field or LatentN_var columns. "
                    f"Columns: {list(frame.columns)}"
                )
    return id_column, variance_columns


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
    grouped["variance_dimensions_used"] = grouped[variance_columns].notna().sum(axis=1)
    return grouped.set_index("TARGETID").to_dict(orient="index")


def lookup_variance(target_id: str, lookup) -> tuple[float, str]:
    if target_id in lookup:
        return lookup[target_id], "exact"
    return {"latent_variance": np.nan, "variance_dimensions_used": 0}, "not_found"


def make_plot(
    data: pd.DataFrame,
    output_path: Path,
    category: str | None = None,
) -> None:
    if category is None:
        usable = data.dropna(subset=["Average", "latent_variance"])
    else:
        usable = data[data["category"] == category].dropna(
            subset=["Average", "latent_variance"]
        )
    fig, ax = plt.subplots(figsize=(8.5, 6))
    categories = sorted(usable["category"].dropna().unique())
    colors = plt.get_cmap("tab10")
    for index, plot_category in enumerate(categories):
        group = usable[usable["category"] == plot_category]
        ax.scatter(
            group["Average"], group["latent_variance"],
            s=34, alpha=0.72, color=colors(index % 10),
            label=f"{plot_category} (n={len(group)})",
        )

    if len(usable) >= 2:
        pearson = usable["Average"].corr(usable["latent_variance"], method="pearson")
        spearman = usable["Average"].rank().corr(usable["latent_variance"].rank())
        label = f"Pearson r={pearson:.2f}; Spearman ρ={spearman:.2f}; n={len(usable)}"
        if (usable["Average"].nunique() > 1
                and usable["latent_variance"].nunique() > 1):
            slope, intercept = np.polyfit(usable["Average"], usable["latent_variance"], 1)
            x_line = np.linspace(usable["Average"].min(), usable["Average"].max(), 100)
            ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.4,
                    label=label)
        else:
            ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top")
    else:
        ax.text(0.03, 0.97, f"Too few matched points to calculate correlation (n={len(usable)})",
                transform=ax.transAxes, va="top")

    ax.set_xlabel("Average visual-inspection score")
    ax.set_ylabel("Mean latent variance across dimensions")
    plot_title = (f"{category}: visual-inspection score vs mean latent variance"
                  if category else
                  "All categories: visual-inspection score vs mean latent variance")
    ax.set_title(plot_title)
    ax.grid(alpha=0.25)
    if categories:
        ax.legend(fontsize="small", loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    combined_path = args.output_dir / "score_vs_latent_variance.png"
    make_plot(results, combined_path)
    plot_paths.append(combined_path)
    for category in sorted(results["category"].dropna().unique()):
        plot_path = args.output_dir / f"score_vs_latent_variance_{category_filename(category)}.png"
        make_plot(results, plot_path, category)
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
