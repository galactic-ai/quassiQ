#!/usr/bin/env python3
"""Rebuild emission/cluster/latent regions with the current N_sigma code.

The emission decision is recomputed from every quality-passing epoch pair and
every emission-line window defined by the imported N_sigma.py.  Targets with
no finite line/pair measurement are recorded as unassessable and are excluded
from the seven Venn regions.

Existing selected_targetids.csv files are never overwritten.  When supplied,
their row order is preserved and invalid IDs are replaced in place in the new
output directory.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import random
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn3_circles
import numpy as np
import pandas as pd


ROOT = Path("/work/11161/kanyuni/ls6/quassiQ_project")
PIPELINE_OUTPUT = ROOT / "pipeline_output"

CATEGORY_ORDER = [
    "emission_only",
    "cluster_only",
    "emission_cluster_only",
    "latent_only",
    "emission_latent_only",
    "cluster_latent_only",
    "all_three",
]

PAIR_COLUMNS = [
    "TARGETID",
    "date_a",
    "date_b",
    "line_name",
    "line_label",
    "line_center_A",
    "window_half_width_A",
    "peak_n_sigma",
    "peak_rest_A",
    "pixels_gt_threshold",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nsigma-script",
        type=Path,
        default=ROOT / "quassiQ/src/pipeline/N_sigma.py",
    )
    parser.add_argument(
        "--cluster-csv",
        type=Path,
        default=Path("/work/10579/prisha/ls6/desi_project/post_clustering.csv"),
    )
    parser.add_argument(
        "--latent-csv",
        type=Path,
        default=PIPELINE_OUTPUT / "latent/latent_targets_with_nonzero_p95_counts.csv",
    )
    parser.add_argument(
        "--final-subset-csv",
        type=Path,
        default=Path("/work/10579/prisha/ls6/desi_project/final_subset.csv"),
    )
    parser.add_argument(
        "--old-category-dir",
        type=Path,
        default=PIPELINE_OUTPUT / "category_plots",
        help="Directory containing the old <category>/selected_targetids.csv files.",
    )
    parser.add_argument(
        "--exclude-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV with category,TARGETID entries that must be replaced "
            "even if the fresh calculation leaves them in the same region."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PIPELINE_OUTPUT / "category_refresh_current_nsigma",
    )
    parser.add_argument("--number", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cluster-date-column", default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace files in --out-dir. Original category_plots files are untouched.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Testing only: evaluate the first N sorted final-subset IDs.",
    )
    args = parser.parse_args()
    if args.number < 1:
        parser.error("--number must be positive")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def import_nsigma(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"N_sigma script not found: {path}")
    spec = importlib.util.spec_from_file_location("current_nsigma", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "EMISSION_LINES",
        "SIGNIFICANCE_THRESHOLD",
        "SNR_CUT",
        "CHI2_MAX",
        "load_candidate_epochs",
        "calculate_n_sigma",
        "calculate_peak_n_sigma",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise AttributeError(f"{path} is missing: {', '.join(missing)}")
    return module


def target_column(frame: pd.DataFrame) -> str:
    for name in ("TARGETID", "TARGET ID"):
        if name in frame.columns:
            return name
    return frame.columns[0]


def read_ids(path: Path, latent: bool = False) -> list[str]:
    frame = pd.read_csv(path, dtype="string")
    column = target_column(frame)
    ids = frame[column].str.strip().replace("", pd.NA)
    if latent and "n_latents_exceed_p95" in frame.columns:
        counts = pd.to_numeric(frame["n_latents_exceed_p95"], errors="coerce")
        ids = ids[counts > 0]
    return ids.dropna().tolist()


def date_key(value) -> str:
    if pd.isna(value):
        raise ValueError("Missing epoch date")
    value = re.sub(r"\.0$", "", str(value).strip())
    return pd.to_datetime(value).strftime("%Y%m%d")


def cluster_jumpers(path: Path, requested_date_column: str | None) -> set[str]:
    frame = pd.read_csv(path, dtype="string")
    if "TARGETID" not in frame.columns or "CLUSTER" not in frame.columns:
        raise ValueError("Cluster CSV must contain TARGETID and CLUSTER")
    date_column = requested_date_column or next(
        (name for name in ("OBS_DATE", "NIGHT_CLEAN", "LASTNIGHT") if name in frame.columns),
        None,
    )
    if date_column is None:
        raise ValueError("Cluster CSV needs OBS_DATE, NIGHT_CLEAN, or LASTNIGHT")

    frame["TARGETID"] = frame["TARGETID"].str.strip()
    jumpers: set[str] = set()
    for target, rows in frame.groupby("TARGETID"):
        assignments: dict[str, str] = {}
        for _, row in rows.iterrows():
            label = row["CLUSTER"]
            if pd.isna(label) or str(label).strip() in {"", "-1", "-1.0"}:
                continue
            label = re.sub(r"\.0+$", "", str(label).strip())
            night = date_key(row[date_column])
            if night in assignments and assignments[night] != label:
                raise ValueError(f"{target}: conflicting cluster labels on {night}")
            assignments[night] = label
        if len(set(assignments.values())) > 1:
            jumpers.add(str(target).strip())
    return jumpers


def quality_epochs(pipeline, epochs: list[dict]) -> list[dict]:
    accepted = []
    for epoch in epochs:
        snr = epoch.get("coadd_snr", np.nan)
        chi2 = epoch.get("chi2_per_pixel", np.nan)
        if not np.isfinite(snr) or snr < pipeline.SNR_CUT:
            continue
        if not np.isfinite(chi2) or chi2 > pipeline.CHI2_MAX:
            continue
        accepted.append(epoch)
    return sorted(accepted, key=lambda epoch: date_key(epoch["date"]))


def evaluate_target(pipeline, target: str) -> tuple[dict, list[dict]]:
    summary = {
        "TARGETID": target,
        "status": "unassessable",
        "failure_reason": "",
        "epochs_total": 0,
        "epochs_after_quality": 0,
        "epoch_pairs": 0,
        "finite_line_pair_measurements": 0,
        "observable_lines": "",
        "max_peak_n_sigma": np.nan,
        "max_pixels_gt_threshold": 0,
        "emission_involved": False,
    }
    rows: list[dict] = []

    try:
        epochs = pipeline.load_candidate_epochs(target)
        summary["epochs_total"] = len(epochs)
        epochs = quality_epochs(pipeline, epochs)
        summary["epochs_after_quality"] = len(epochs)
        if len(epochs) < 2:
            summary["failure_reason"] = "Fewer than two epochs pass current S/N and chi2 cuts"
            return summary, rows

        peaks: list[float] = []
        observable: set[str] = set()
        for epoch_a, epoch_b in combinations(epochs, 2):
            summary["epoch_pairs"] += 1
            calc = pipeline.calculate_n_sigma(epoch_a, epoch_b)
            wave = np.asarray(calc["wave"], dtype=float)
            n_sigma = np.asarray(calc["n_sigma"], dtype=float)

            for line_name, config in pipeline.EMISSION_LINES.items():
                center = float(config["wavelength"])
                window = float(config["window"])
                peak, peak_wave = pipeline.calculate_peak_n_sigma(
                    wave, n_sigma, center, window
                )
                if not np.isfinite(peak):
                    continue

                in_window = (
                    np.isfinite(wave)
                    & np.isfinite(n_sigma)
                    & (wave >= center - window)
                    & (wave <= center + window)
                )
                pixel_count = int(
                    np.count_nonzero(in_window & (n_sigma > pipeline.SIGNIFICANCE_THRESHOLD))
                )
                observable.add(line_name)
                peaks.append(float(peak))
                summary["max_pixels_gt_threshold"] = max(
                    summary["max_pixels_gt_threshold"], pixel_count
                )
                rows.append(
                    {
                        "TARGETID": target,
                        "date_a": date_key(epoch_a["date"]),
                        "date_b": date_key(epoch_b["date"]),
                        "line_name": line_name,
                        "line_label": config["label"],
                        "line_center_A": center,
                        "window_half_width_A": window,
                        "peak_n_sigma": float(peak),
                        "peak_rest_A": float(peak_wave),
                        "pixels_gt_threshold": pixel_count,
                    }
                )

        summary["finite_line_pair_measurements"] = len(rows)
        summary["observable_lines"] = ";".join(sorted(observable))
        if not peaks:
            summary["failure_reason"] = "No finite N_sigma values in current emission windows"
            return summary, rows

        summary["max_peak_n_sigma"] = max(peaks)
        summary["emission_involved"] = bool(
            summary["max_peak_n_sigma"] > pipeline.SIGNIFICANCE_THRESHOLD
        )
        summary["status"] = "assessable"
        return summary, rows
    except Exception as exc:
        summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return summary, rows


def exclusive_regions(
    emission: set[str], cluster: set[str], latent: set[str]
) -> dict[str, set[str]]:
    return {
        "emission_only": emission - cluster - latent,
        "cluster_only": cluster - emission - latent,
        "emission_cluster_only": (emission & cluster) - latent,
        "latent_only": latent - emission - cluster,
        "emission_latent_only": (emission & latent) - cluster,
        "cluster_latent_only": (cluster & latent) - emission,
        "all_three": emission & cluster & latent,
    }


def make_venn(regions: dict[str, set[str]], threshold: float, output: Path) -> None:
    subsets = (
        len(regions["emission_only"]),
        len(regions["cluster_only"]),
        len(regions["emission_cluster_only"]),
        len(regions["latent_only"]),
        len(regions["emission_latent_only"]),
        len(regions["cluster_latent_only"]),
        len(regions["all_three"]),
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    diagram = venn3(
        subsets=subsets,
        set_labels=(
            f"Emission windows\nany pair peak N_sigma > {threshold:g}",
            "Cluster Jumpers",
            "Latent Outliers\nn_latents_exceed_p95 > 0",
        ),
        set_colors=("#2f3061", "#4ecdc4", "#ffe66d"),
        alpha=0.65,
        ax=ax,
    )
    venn3_circles(subsets=subsets, linewidth=1.5, color="black", ax=ax)
    for label in diagram.subset_labels:
        if label is not None:
            label.set_fontsize(11)
            label.set_bbox(
                dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.0)
            )
    ax.set_title("Unique TARGETID overlap using current N_sigma")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fresh_selection_and_replacements(
    regions: dict[str, set[str]],
    old_dir: Path,
    out_dir: Path,
    number: int,
    seed: int,
    forced_exclusions: dict[str, set[str]],
) -> pd.DataFrame:
    target_category = {
        target: category
        for category, members in regions.items()
        for target in members
    }
    replacement_rows: list[dict] = []

    for category_index, category in enumerate(CATEGORY_ORDER):
        members = regions[category]
        excluded = forced_exclusions.get(category, set())
        selectable_members = members - excluded
        category_dir = out_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"TARGETID": sorted(members)}).to_csv(
            category_dir / "all_category_targetids.csv", index=False
        )

        old_path = old_dir / category / "selected_targetids.csv"
        desired = min(number, len(selectable_members))
        rng = random.Random(seed + category_index)

        if old_path.is_file():
            old = read_ids(old_path)
            old = old[:number]
            seen_old: set[str] = set()
            invalid = []
            for target in old:
                if target not in selectable_members or target in seen_old:
                    invalid.append(target)
                seen_old.add(target)
            pool = sorted(selectable_members - set(old))
            if len(pool) < len(invalid):
                raise RuntimeError(
                    f"{category}: need {len(invalid)} replacements but only "
                    f"{len(pool)} unused fresh members exist"
                )
            replacements = iter(rng.sample(pool, len(invalid)))
            refreshed = []
            used = set()
            for row_number, target in enumerate(old, start=1):
                if target in selectable_members and target not in used:
                    refreshed.append(target)
                    used.add(target)
                    continue
                replacement = next(replacements)
                refreshed.append(replacement)
                used.add(replacement)
                replacement_rows.append(
                    {
                        "category": category,
                        "row": row_number,
                        "old_TARGETID": target,
                        "suggested_replacement_TARGETID": replacement,
                        "old_target_fresh_category": target_category.get(
                            target, "unassessable_or_outside_fresh_regions"
                        ),
                        "reason": (
                            "explicitly excluded as a known anomaly"
                            if target in excluded
                            else "old ID no longer belongs to this freshly recomputed region"
                        ),
                    }
                )

            if len(refreshed) < desired:
                extra_pool = sorted(selectable_members - set(refreshed))
                extra = rng.sample(extra_pool, desired - len(refreshed))
                refreshed.extend(extra)
        else:
            refreshed = rng.sample(sorted(selectable_members), desired)

        pd.DataFrame({"TARGETID": refreshed[:desired]}).to_csv(
            category_dir / "selected_targetids.csv", index=False
        )

    replacements = pd.DataFrame(
        replacement_rows,
        columns=[
            "category",
            "row",
            "old_TARGETID",
            "suggested_replacement_TARGETID",
            "old_target_fresh_category",
            "reason",
        ],
    )
    replacements.to_csv(out_dir / "replacement_candidates.csv", index=False)
    return replacements


def read_forced_exclusions(path: Path | None) -> dict[str, set[str]]:
    exclusions = {category: set() for category in CATEGORY_ORDER}
    if path is None:
        return exclusions
    frame = pd.read_csv(path, dtype="string")
    required = {"category", "TARGETID"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} must contain category and TARGETID")
    for _, row in frame.iterrows():
        category = str(row["category"]).strip()
        target = str(row["TARGETID"]).strip()
        if category not in exclusions:
            raise ValueError(f"Unknown exclusion category: {category}")
        if target and target.lower() != "nan":
            exclusions[category].add(target)
    return exclusions


def main() -> None:
    args = parse_args()
    pipeline = import_nsigma(args.nsigma_script)

    key_outputs = [
        args.out_dir / "fresh_target_nsigma_summary.csv",
        args.out_dir / "fresh_pair_line_nsigma.csv",
        args.out_dir / "replacement_candidates.csv",
    ]
    existing = [path for path in key_outputs if path.exists()]
    if existing and not args.overwrite:
        names = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Output already exists. Use --overwrite or choose a new --out-dir:\n" + names
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    final_ids = sorted(set(read_ids(args.final_subset_csv)))
    if args.limit is not None:
        final_ids = final_ids[: args.limit]
        print(f"TEST MODE: limited to {len(final_ids)} targets", flush=True)
    final_set = set(final_ids)
    raw_cluster = cluster_jumpers(args.cluster_csv, args.cluster_date_column)
    raw_latent = set(read_ids(args.latent_csv, latent=True))

    summaries: list[dict] = []
    with (args.out_dir / "fresh_pair_line_nsigma.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_COLUMNS)
        writer.writeheader()
        for index, target in enumerate(final_ids, start=1):
            summary, rows = evaluate_target(pipeline, target)
            summaries.append(summary)
            writer.writerows(rows)
            print(
                f"[{index}/{len(final_ids)}] {target}: {summary['status']} "
                f"pairs={summary['epoch_pairs']} "
                f"max_peak={summary['max_peak_n_sigma']} "
                f"emission={summary['emission_involved']}",
                flush=True,
            )

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(
        args.out_dir / "fresh_target_nsigma_summary.csv", index=False
    )
    assessable = set(
        summary_frame.loc[summary_frame["status"] == "assessable", "TARGETID"].astype(str)
    )
    emission = set(
        summary_frame.loc[
            (summary_frame["status"] == "assessable")
            & summary_frame["emission_involved"].astype(bool),
            "TARGETID",
        ].astype(str)
    )
    eligible = final_set & assessable
    cluster = raw_cluster & eligible
    latent = raw_latent & eligible
    regions = exclusive_regions(emission, cluster, latent)

    pd.DataFrame(
        [
            {"TARGETID": target, "category": category}
            for category in CATEGORY_ORDER
            for target in sorted(regions[category])
        ]
    ).to_csv(args.out_dir / "all_categories_targetids.csv", index=False)

    pd.DataFrame(
        [
            {"category": category, "target_count": len(regions[category])}
            for category in CATEGORY_ORDER
        ]
        + [
            {"category": "unassessable", "target_count": len(final_set - assessable)}
        ]
    ).to_csv(args.out_dir / "category_summary.csv", index=False)

    make_venn(
        regions,
        pipeline.SIGNIFICANCE_THRESHOLD,
        args.out_dir / "emission_cluster_latent_venn_current_nsigma.png",
    )
    replacements = fresh_selection_and_replacements(
        regions,
        args.old_category_dir,
        args.out_dir,
        args.number,
        args.seed,
        read_forced_exclusions(args.exclude_csv),
    )

    print("\nFresh category counts:")
    for category in CATEGORY_ORDER:
        print(f"  {category}: {len(regions[category])}")
    print(f"  unassessable: {len(final_set - assessable)}")
    print(f"Replacement suggestions: {len(replacements)}")
    print(f"Saved everything under: {args.out_dir}")


if __name__ == "__main__":
    main()
