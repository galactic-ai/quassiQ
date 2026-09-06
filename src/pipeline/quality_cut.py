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
6. Per-coadd chi2 cut   : reduced chi2 <= CHI2_CUT (checked only if S/N passes)
7. Minimum kept coadds  : keep TARGETIDs with >= MIN_KEPT_COADDS coadds
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
CHI2_CUT = 10.0
MIN_KEPT_COADDS = 2


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


def apply_snr_chi2_cut(df, coadd_root, plot_root,
                        snr_cut=SNR_CUT, chi2_cut=CHI2_CUT, min_kept=MIN_KEPT_COADDS):
    """Stages 5-7: per-coadd S/N + chi2 classification, then a minimum-kept-coadds cut."""
    allowed_ids = set(df["TARGETID"].astype(str).unique())

    snr_ok_map, stage_stats = screen_targets_by_snr(
        coadd_root=coadd_root,
        plot_root=plot_root,
        allowed_target_ids=allowed_ids,
        snr_cut=snr_cut,
        chi2_cut=chi2_cut,
        min_kept=min_kept,
    )

    print(f"Targets scanned in S/N + chi2 stage:    {stage_stats['targets_scanned']}")
    print(f"Targets passing S/N + chi2 stage:       {stage_stats['targets_with_min_kept']}")
    print(f"Targets removed (too few kept coadds):  {stage_stats['targets_removed_lt_min_kept']}")

    kept_ids = set(snr_ok_map.keys())
    df = df[df["TARGETID"].astype(str).isin(kept_ids)]
    print(f"Rows after S/N + chi2 filtering: {len(df)}")
    return df, snr_ok_map


def main():
    df = load_catalog(FITS_CATALOG)
    df = apply_quality_flags(df)
    df = apply_redshift_window(df)
    df = apply_repeat_observation_filter(df)
    df = apply_duration_filter(df)
    df, snr_ok_map = apply_snr_chi2_cut(df, COADD_ROOT, PLOT_ROOT)

    df = df.sort_values(["duration_days", "TARGETID"], ascending=[False, True])
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSummary")
    print(f"Saved final candidates to: {OUTPUT_CSV}")
    print(f"Final remaining rows: {len(df)}")
    print(f"Final unique TARGETIDs: {df['TARGETID'].nunique()}")


if __name__ == "__main__":
    main()
