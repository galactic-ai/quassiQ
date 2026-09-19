#!/usr/bin/env python3
"""

python /work/11161/kanyuni/ls6/quassiQ_project/quassiQ/src/pipeline/plot_target_list_overlap.py \
  --latent-csv /work/11161/kanyuni/ls6/quassiQ_project/latent/latent_std_normalized_p95_counts_by_target.csv \
  --output-prefix /work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/overlap/target_list_overlap

Plot overlap among emission-line candidates and a latent-variability list.

A target is an emission-line candidate when ``peak_n_sigma > 3`` in at least
one of the Ly-alpha, C IV, or Mg II result files. A target is a latent candidate
when ``n_latents_exceed_p95 >= 1``.

The script writes:
  1. an UpSet-style PNG of the exact three-line intersections;
  2. a CSV containing the emission-candidate union and all membership flags.

TARGETID is read as a string so large DESI identifiers are never rounded.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/work/11161/kanyuni/ls6/quassiQ_project")

DEFAULT_LINE_FILES = {
    "Lyα": PROJECT_ROOT / "pipeline_output/lya/batch/lya_first_1000_results.csv",
    "C IV": PROJECT_ROOT / "pipeline_output/civ/batch/civ_first_1000_results.csv",
    "Mg II": PROJECT_ROOT / "pipeline_output/mgii/batch/mgii_first_1000_results.csv",
}

# Change this default if the latent CSV lives elsewhere, or pass --latent-csv.
DEFAULT_LATENT_FILE = PROJECT_ROOT / "latent_std_normalized_p95_counts_by_target.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lya-csv", type=Path, default=DEFAULT_LINE_FILES["Lyα"])
    parser.add_argument("--civ-csv", type=Path, default=DEFAULT_LINE_FILES["C IV"])
    parser.add_argument("--mgii-csv", type=Path, default=DEFAULT_LINE_FILES["Mg II"])
    parser.add_argument("--latent-csv", type=Path, default=DEFAULT_LATENT_FILE)
    parser.add_argument("--nsigma-threshold", type=float, default=3.0)
    parser.add_argument("--latent-threshold", type=int, default=1)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("target_list_overlap"),
        help="Output path without extension (default: target_list_overlap).",
    )
    return parser.parse_args()


def read_emission_file(path: Path, expected_line: str, threshold: float) -> tuple[set[str], pd.Series]:
    """Return significant TARGETIDs and maximum peak_n_sigma per target."""
    required = {"TARGETID", "peak_n_sigma"}
    df = pd.read_csv(path, dtype={"TARGETID": "string"})
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")

    # This catches accidentally copied or misnamed line-result files.
    if "line_name" in df.columns:
        found = set(df["line_name"].dropna().astype(str).str.lower().unique())
        accepted = {
            "Lyα": {"lya", "lyalpha", "ly_alpha", "ly-a"},
            "C IV": {"civ", "c_iv", "c iv"},
            "Mg II": {"mgii", "mg_ii", "mg ii"},
        }[expected_line]
        if found and found.isdisjoint(accepted):
            warnings.warn(
                f"{path.name} was supplied as {expected_line}, but line_name contains {sorted(found)}."
            )

    df["peak_n_sigma"] = pd.to_numeric(df["peak_n_sigma"], errors="coerce")
    max_sigma = df.groupby("TARGETID", dropna=True)["peak_n_sigma"].max()
    selected = set(max_sigma.index[max_sigma > threshold].astype(str))
    return selected, max_sigma


def read_latent_file(path: Path, threshold: int) -> tuple[set[str], pd.Series]:
    """Return latent-candidate TARGETIDs and maximum count per target."""
    required = {"TARGETID", "n_latents_exceed_p95"}
    df = pd.read_csv(path, dtype={"TARGETID": "string"})
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required column(s): {sorted(missing)}")
    df["n_latents_exceed_p95"] = pd.to_numeric(
        df["n_latents_exceed_p95"], errors="coerce"
    )
    max_count = df.groupby("TARGETID", dropna=True)["n_latents_exceed_p95"].max()
    selected = set(max_count.index[max_count >= threshold].astype(str))
    return selected, max_count


def exact_line_intersections(line_sets: dict[str, set[str]]) -> list[tuple[tuple[bool, ...], int]]:
    """Count mutually exclusive intersections of the three emission sets."""
    labels = list(line_sets)
    union = set().union(*line_sets.values())
    counts: dict[tuple[bool, ...], int] = {}
    for target_id in union:
        pattern = tuple(target_id in line_sets[label] for label in labels)
        counts[pattern] = counts.get(pattern, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def build_membership_table(
    line_sets: dict[str, set[str]],
    sigma_by_line: dict[str, pd.Series],
    latent_set: set[str],
    latent_counts: pd.Series,
) -> pd.DataFrame:
    """Build one row per target selected by at least one emission line."""
    emission_union = sorted(set().union(*line_sets.values()), key=int)
    out = pd.DataFrame({"TARGETID": pd.Series(emission_union, dtype="string")})
    for label, column in [("Lyα", "in_lya"), ("C IV", "in_civ"), ("Mg II", "in_mgii")]:
        out[column] = out["TARGETID"].isin(line_sets[label])
        out[f"{column.removeprefix('in_')}_peak_n_sigma"] = out["TARGETID"].map(
            sigma_by_line[label]
        )
    out["n_emission_lines_over_threshold"] = out[["in_lya", "in_civ", "in_mgii"]].sum(axis=1)
    out["in_latent_list"] = out["TARGETID"].isin(latent_set)
    out["n_latents_exceed_p95"] = out["TARGETID"].map(latent_counts)
    return out


def add_count_labels(ax: plt.Axes, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            (bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def make_plot(
    line_sets: dict[str, set[str]],
    latent_set: set[str],
    intersections: list[tuple[tuple[bool, ...], int]],
    output_png: Path,
    nsigma_threshold: float,
    latent_threshold: int,
) -> None:
    """Create an UpSet-style line-overlap plot plus latent comparison."""
    labels = list(line_sets)
    patterns = [pattern for pattern, _ in intersections]
    counts = [count for _, count in intersections]
    emission_union = set().union(*line_sets.values())
    both = len(emission_union & latent_set)
    emission_only = len(emission_union - latent_set)
    latent_fraction = 100 * both / len(emission_union) if emission_union else 0.0

    fig = plt.figure(figsize=(12.5, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[3.2, 1.25], width_ratios=[2.1, 1.0])
    ax_counts = fig.add_subplot(grid[0, 0])
    ax_matrix = fig.add_subplot(grid[1, 0], sharex=ax_counts)
    ax_latent = fig.add_subplot(grid[:, 1])

    x = np.arange(len(counts))
    bars = ax_counts.bar(x, counts, color="#355C7D", width=0.72)
    add_count_labels(ax_counts, bars)
    ax_counts.set_ylabel("Number of targets")
    ax_counts.set_title("Exact overlap among emission-line candidates")
    ax_counts.spines[["top", "right"]].set_visible(False)
    ax_counts.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_counts.grid(axis="y", alpha=0.2)

    colors = {"Lyα": "#5B8FF9", "C IV": "#61DDAA", "Mg II": "#F6BD16"}
    for col, pattern in enumerate(patterns):
        active_rows = [row for row, included in enumerate(pattern) if included]
        ax_matrix.scatter(
            [col] * len(labels), range(len(labels)), s=52, color="#D5D9DE", zorder=2
        )
        if len(active_rows) > 1:
            ax_matrix.plot([col, col], [min(active_rows), max(active_rows)], color="#333333", lw=2)
        for row in active_rows:
            ax_matrix.scatter(col, row, s=72, color=colors[labels[row]], edgecolor="white", zorder=3)
    ax_matrix.set_yticks(range(len(labels)), labels)
    ax_matrix.set_xticks(x, [str(i + 1) for i in x])
    ax_matrix.set_xlabel("Exclusive intersection (sorted by count)")
    ax_matrix.set_ylim(len(labels) - 0.5, -0.5)
    ax_matrix.spines[:].set_visible(False)
    ax_matrix.tick_params(length=0)

    latent_bars = ax_latent.bar(
        ["Also latent", "Not latent"],
        [both, emission_only],
        color=["#8E5EA2", "#BFC5CC"],
        width=0.62,
    )
    add_count_labels(ax_latent, latent_bars)
    ax_latent.set_ylabel("Number of emission candidates")
    ax_latent.set_title(f"Emission union vs. n_latents_exceed_p95 ≥ {latent_threshold}")
    ax_latent.text(
        0.5,
        0.96,
        f"{both:,} / {len(emission_union):,} = {latent_fraction:.1f}% overlap",
        transform=ax_latent.transAxes,
        ha="center",
        va="top",
        fontsize=11,
    )
    ax_latent.spines[["top", "right"]].set_visible(False)
    ax_latent.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Candidate overlap: peak Nσ > {nsigma_threshold:g} in ≥1 emission line",
        fontsize=15,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    line_paths = {"Lyα": args.lya_csv, "C IV": args.civ_csv, "Mg II": args.mgii_csv}

    line_sets: dict[str, set[str]] = {}
    sigma_by_line: dict[str, pd.Series] = {}
    for label, path in line_paths.items():
        line_sets[label], sigma_by_line[label] = read_emission_file(
            path, label, args.nsigma_threshold
        )
    latent_set, latent_counts = read_latent_file(args.latent_csv, args.latent_threshold)

    membership = build_membership_table(line_sets, sigma_by_line, latent_set, latent_counts)
    intersections = exact_line_intersections(line_sets)

    output_png = args.output_prefix.with_suffix(".png")
    output_csv = args.output_prefix.with_name(args.output_prefix.name + "_membership.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(output_csv, index=False)
    make_plot(
        line_sets,
        latent_set,
        intersections,
        output_png,
        args.nsigma_threshold,
        args.latent_threshold,
    )

    emission_union = set(membership["TARGETID"].astype(str))
    overlap = emission_union & latent_set
    print(f"Lyα candidates:  {len(line_sets['Lyα']):,}")
    print(f"C IV candidates: {len(line_sets['C IV']):,}")
    print(f"Mg II candidates:{len(line_sets['Mg II']):,}")
    print(f"Emission union:   {len(emission_union):,}")
    print(f"Also latent:      {len(overlap):,} ({len(overlap) / len(emission_union):.1%})" if emission_union else "Also latent: 0")
    print(f"Saved plot:       {output_png}")
    print(f"Saved membership: {output_csv}")


if __name__ == "__main__":
    main()
