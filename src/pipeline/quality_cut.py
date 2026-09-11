#!/usr/bin/env python
# coding: utf-8
"""
Quality cuts for the QSO catalog -> CLQ_candidates.csv

Assumes the following directory layout:

    ~/ls6/quassiQ_project/              (PROJECT_ROOT)
    |-- quassiQ/                        (git clone, QUASSIQ_ROOT)
    |   `-- src/pipeline/quality_cut.py (this script)
    `-- coadds/                         (COADD_ROOT)

Stages
------
1. Quality flags       : ZWARN in {0, 4}, COADD_FIBERSTATUS == 0
2. Redshift window      : Z_MIN <= Z <= Z_MAX
3. Repeat observations  : keep TARGETIDs with >1 remaining observation
4. Duration             : keep TARGETIDs spanning >= MIN_DURATION_DAYS
5. Per-coadd S/N cut    : median S/N >= SNR_CUT
6. Minimum kept coadds  : keep TARGETIDs with >= MIN_K EPT_COADDS coadds
   surviving stages 5-6

"""

from pathlib import Path

import pandas as pd
from astropy.io import fits
import numpy as np

from fits_qc_utils import screen_targets_by_snr


THIS_FILE = Path(__file__).resolve()
QUASSIQ_ROOT = THIS_FILE.parents[2]      # .../quassiQ_project/quassiQ
PROJECT_ROOT = THIS_FILE.parents[3]      # .../quassiQ_project

FITS_CATALOG = PROJECT_ROOT / "QSO_iron" / "iron" / "QSO_cat_iron_cumulative_v0.fits"
COADD_ROOT = PROJECT_ROOT / "coadds"
PLOT_ROOT = PROJECT_ROOT / "quality_cut_diagnostics"  # per-target median-S/N CSVs
OUTPUT_CSV = "CLQ_candidates.csv"

Z_MIN, Z_MAX = 2.1, 3.5
MIN_DURATION_DAYS = 30
SNR_CUT = 2.0
MIN_KEPT_COADDS = 2

def report_filter(stage_name, before_df, after_df):
    before_rows = len(before_df)
    after_rows = len(after_df)

    before_targets = before_df["TARGETID"].nunique()
    after_targets = after_df["TARGETID"].nunique()

    print(f"\n{stage_name}")
    print(f"  Rows before:        {before_rows}")
    print(f"  Rows after:         {after_rows}")
    print(f"  Rows removed:       {before_rows - after_rows}")
    print(f"  TARGETIDs before:   {before_targets}")
    print(f"  TARGETIDs after:    {after_targets}")
    print(f"  TARGETIDs removed:  {before_targets - after_targets}")

def load_catalog(fits_path):
    with fits.open(fits_path) as hdul:
        # prevents potential FITS byte-order/endianness issues in pandas
        data = np.ascontiguousarray(hdul[1].data)
    df = pd.DataFrame(data)
    print(f"Initial row count: {len(df)}")
    return df


def apply_quality_flags(df):
    df = df[df["ZWARN"].isin([0, 4])]
    df = df[df["COADD_FIBERSTATUS"] == 0]
    print(f"Rows after quality filtering: {len(df)}")
    return df


def apply_redshift_window(df, z_min=Z_MIN, z_max=Z_MAX):
    mask_z = (df["Z"] >= z_min) & (df["Z"] <= z_max)
    valid_ids = df.loc[mask_z, "TARGETID"].unique()
    df = df[df["TARGETID"].isin(valid_ids)]
    print(f"Rows after redshift filtering ({z_min} <= Z <= {z_max}): {len(df)}")
    return df


def apply_repeat_observation_filter(df):
    df = df[df["TARGETID"].duplicated(keep=False)]
    print(f"Rows with multiple observations: {len(df)}")
    return df


def apply_duration_filter(df, min_days=MIN_DURATION_DAYS):
    df = df.copy()
    df["LASTNIGHT"] = pd.to_datetime(df["LASTNIGHT"], format="%Y%m%d")

    duration = df.groupby("TARGETID")["LASTNIGHT"].agg(["min", "max"])
    duration["duration_days"] = (duration["max"] - duration["min"]).dt.days

    df = df.merge(duration["duration_days"], on="TARGETID")

    valid_ids = df.loc[df["duration_days"] >= min_days, "TARGETID"].unique()
    df = df[df["TARGETID"].isin(valid_ids)]
    print(f"Rows after duration filtering (>= {min_days} days): {len(df)}")
    return df

def apply_snr_cut(
    df,
    coadd_root,
    plot_root,
    snr_cut=SNR_CUT,
    min_kept=MIN_KEPT_COADDS,
):
    """Apply the per-coadd S/N and minimum-kept-coadds cuts."""

    allowed_ids = set(df["TARGETID"].astype(str).unique())

    snr_ok_map, stage_stats = screen_targets_by_snr(
        coadd_root=coadd_root,
        plot_root=plot_root,
        allowed_target_ids=allowed_ids,
        snr_cut=snr_cut,
        min_kept=min_kept,
    )

    print("\nPer-coadd S/N details")
    print(
        f"  Targets with coadd directories: "
        f"{stage_stats['targets_scanned']}"
    )
    print(
        f"  Coadds passing S/N: "
        f"{stage_stats['total_kept_coadds']}"
    )
    print(
        f"  Coadds rejected for low S/N: "
        f"{stage_stats['total_rejected_low']}"
    )
    print(
        f"  Coadds with invalid S/N: "
        f"{stage_stats['total_invalid']}"
    )
    print(
        f"  Targets with >= {min_kept} passing coadds: "
        f"{stage_stats['targets_with_min_kept']}"
    )
    print(
        f"  Targets with < {min_kept} passing coadds: "
        f"{stage_stats['targets_removed_lt_min_kept']}"
    )

    kept_ids = set(snr_ok_map)
    filtered_df = df[
        df["TARGETID"].astype(str).isin(kept_ids)
    ].copy()

    return filtered_df, snr_ok_map

def main():
    df = load_catalog(FITS_CATALOG)

    print("\nStage 0: Initial catalog")
    print(f"  Rows:      {len(df)}")
    print(f"  TARGETIDs: {df['TARGETID'].nunique()}")

    before = df
    df = apply_quality_flags(df)
    report_filter(
        "Stage 1: Quality flags",
        before,
        df,
    )

    before = df
    df = apply_redshift_window(df)
    report_filter(
        f"Stage 2: Redshift ({Z_MIN} <= Z <= {Z_MAX})",
        before,
        df,
    )

    before = df
    df = apply_repeat_observation_filter(df)
    report_filter(
        "Stage 3: Repeat observations",
        before,
        df,
    )

    before = df
    df = apply_duration_filter(df)
    report_filter(
        f"Stage 4: Duration >= {MIN_DURATION_DAYS} days",
        before,
        df,
    )

    before = df
    df, snr_ok_map = apply_snr_cut(
        df,
        COADD_ROOT,
        PLOT_ROOT,
    )
    report_filter(
        (
            f"Stages 5 and 7: Median S/N >= {SNR_CUT} "
            f"and at least {MIN_KEPT_COADDS} passing coadds"
        ),
        before,
        df,
    )

    df = df.sort_values(
        ["duration_days", "TARGETID"],
        ascending=[False, True],
    )

    df.to_csv(OUTPUT_CSV, index=False)

    target_output = "CLQ_candidate_TARGETIDs.csv"

    unique_targets = (
        df[["TARGETID"]]
        .drop_duplicates()
        .sort_values("TARGETID")
    )

    unique_targets.to_csv(
        target_output,
        index=False,
    )

    print("\n========== FINAL SUMMARY ==========")
    print(f"Final catalog rows:        {len(df)}")
    print(f"Final unique TARGETIDs:    {df['TARGETID'].nunique()}")
    print(f"Full output:               {OUTPUT_CSV}")
    print(f"Unique TARGETID output:    {target_output}")
    print("===================================")


if __name__ == "__main__":
    main()
