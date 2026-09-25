#!/usr/bin/env python3
"""Plot unique CLQ candidate counts per emission line (peak_n_sigma > 3)."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output")
LINES = {
    "lya": "Lyα",
    "nv": "N V",
    "siiv_oiv": "Si IV + O IV]",
    "civ": "C IV",
    "heii": "He II",
    "oiii": "O III]",
    "aliii": "Al III",
    "ciii": "C III]",
    "mgii": "Mg II",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=ROOT / "csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "n_sigma")
    parser.add_argument("--threshold", type=float, default=3.0)
    args = parser.parse_args()
    if not np.isfinite(args.threshold) or args.threshold < 0:
        parser.error("--threshold must be finite and nonnegative")

    rows = []
    for line, label in LINES.items():
        path = args.csv_dir / f"{line}_all_results.csv"
        df = pd.read_csv(path, dtype={"TARGETID": "string"})
        required = {"TARGETID", "peak_n_sigma"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path}: missing columns {sorted(required - set(df.columns))}")

        ids = df["TARGETID"].str.strip().replace("", pd.NA)
        nsigma = pd.to_numeric(df["peak_n_sigma"], errors="coerce")
        valid = ids.notna() & np.isfinite(nsigma)
        if "status" in df.columns:
            valid &= df["status"].eq("success")
        selected = valid & (nsigma > args.threshold)
        rows.append({
            "line_name": line,
            "line_label": label,
            "n_clq_candidates": ids[selected].nunique(),
            "n_valid_targets": ids[valid].nunique(),
        })

    counts = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "clq_counts_by_line.csv"
    plot_path = args.out_dir / "clq_counts_by_line.png"
    counts.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(counts["line_label"], counts["n_clq_candidates"], color="steelblue")
    ax.bar_label(bars, padding=4, fmt="%d")
    ax.set_ylabel("Number of unique CLQ candidates")
    ax.set_xlabel("Emission line")
    ax.set_title(f"CLQ candidates by emission line (peak Nσ > {args.threshold:g})")
    ax.set_ylim(0, max(1, counts["n_clq_candidates"].max()) * 1.18)
    ax.tick_params(axis="x", rotation=25)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.02, "Targets can appear in multiple bars; line coverage and valid sample sizes differ.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(counts.to_string(index=False))
    print(f"\nSaved: {plot_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
