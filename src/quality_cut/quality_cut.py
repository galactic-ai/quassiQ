#!/usr/bin/env python
# coding: utf-8
"""Apply catalog-level and per-coadd quality cuts to the DESI QSO catalog.

Cuts are applied in this order:

1. Require ZWARN == 0 and COADD_FIBERSTATUS == 0.
2. Require at least one remaining observation of a TARGETID to have
   2.1 <= Z <= 3.5. Keep all observations belonging to each qualifying target.
3. Require more than one observation per TARGETID.
4. Require a time baseline of at least 7 days.
5. Calculate the median per-pixel S/N of each corresponding original coadd.
6. Keep coadds with median S/N >= 2 and TARGETIDs with at least two such coadds.

The script writes a row-level candidate catalog, a unique-TARGETID catalog,
and an S/N diagnostic table containing every coadd examined at stage 5.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
from astropy.io import fits

THIS_FILE = Path(__file__).resolve()
QUASSIQ_ROOT = THIS_FILE.parents[2]
PROJECT_ROOT = THIS_FILE.parents[3]

DEFAULT_FITS_CATALOG = (
    PROJECT_ROOT
    / "QSO_iron"
    / "iron"
    / "QSO_cat_iron_cumulative_v0.fits"
)

DEFAULT_COADD_ROOT = Path("/work/10579/prisha/ls6/desi_project/output_coadds")
OUTPUT_ROOT = PROJECT_ROOT / "quality_cut"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_OUTPUT_CSV = (OUTPUT_ROOT / "CLQ_candidates_7days_merged.csv")
DEFAULT_TARGET_OUTPUT = (OUTPUT_ROOT / "CLQ_candidate_TARGETIDs_7days_merged.csv")
DEFAULT_SNR_DIAGNOSTICS = (OUTPUT_ROOT / "CLQ_coadd_snr_diagnostics_7days_merged.csv")

Z_MIN = 2.1
Z_MAX = 3.5
MIN_DURATION_DAYS = 7
SNR_CUT = 2.0
MIN_KEPT_COADDS = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply catalog and per-coadd S/N cuts to the DESI QSO catalog."
    )
    parser.add_argument("--fits-catalog", type=Path, default=DEFAULT_FITS_CATALOG)
    parser.add_argument("--coadd-root", type=Path, default=DEFAULT_COADD_ROOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--target-output", type=Path, default=DEFAULT_TARGET_OUTPUT)
    parser.add_argument(
        "--snr-diagnostics", type=Path, default=DEFAULT_SNR_DIAGNOSTICS
    )
    return parser.parse_args()


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
    if not fits_path.is_file():
        raise FileNotFoundError(f"Catalog does not exist: {fits_path}")

    with fits.open(fits_path, memmap=True) as hdul:
        data = np.ascontiguousarray(hdul[1].data)

    df = pd.DataFrame(data)
    print(f"Initial row count: {len(df)}")
    return df


def apply_quality_flags(df):
    return df.loc[
        (df["ZWARN"] == 0) & (df["COADD_FIBERSTATUS"] == 0)
    ].copy()


def apply_redshift_window(df, z_min=Z_MIN, z_max=Z_MAX):
    """Keep all rows of targets having at least one observation in range."""
    row_in_range = df["Z"].between(z_min, z_max, inclusive="both")
    any_obs_in_range = row_in_range.groupby(df["TARGETID"]).transform("any")
    return df.loc[any_obs_in_range].copy()


def apply_repeat_observation_filter(df):
    return df.loc[df["TARGETID"].duplicated(keep=False)].copy()


def apply_duration_filter(df, min_days=MIN_DURATION_DAYS):
    df = df.copy()
    df["LASTNIGHT"] = pd.to_datetime(
        df["LASTNIGHT"].astype(str), format="%Y%m%d", errors="raise"
    )

    duration_days = df.groupby("TARGETID")["LASTNIGHT"].transform(
        lambda dates: (dates.max() - dates.min()).days
    )
    df["duration_days"] = duration_days
    return df.loc[df["duration_days"] >= min_days].copy()


def coadd_path_for_row(row, coadd_root):
    target_id = int(row["TARGETID"])
    tile_id = int(row["TILEID"])
    petal_loc = int(row["PETAL_LOC"])
    night = pd.Timestamp(row["LASTNIGHT"]).strftime("%Y%m%d")
    filename = f"coadd-{petal_loc}-{tile_id}-{night}-{target_id}.fits"
    return coadd_root / str(target_id) / filename


def get_target_row_idx(hdul, target_id):
    """Return the target's FIBERMAP row, or None if it cannot be identified."""
    if "FIBERMAP" not in hdul:
        return None

    fibermap = hdul["FIBERMAP"].data
    if getattr(fibermap, "names", None) is None or "TARGETID" not in fibermap.names:
        return None

    matches = np.flatnonzero(
        np.asarray(fibermap["TARGETID"]).astype(str) == str(target_id)
    )
    return int(matches[0]) if matches.size else None


def select_spectrum(array, row_idx):
    """Select one spectrum from a DESI (nspec, npixel) image array."""
    array = np.asarray(array)
    if array.ndim == 1:
        return array
    if array.ndim != 2 or row_idx >= array.shape[0]:
        raise IndexError(
            f"Cannot select spectrum row {row_idx} from array shape {array.shape}"
        )
    return array[row_idx]


def calculate_median_snr(coadd_path, target_id):
    """Return median Flux*sqrt(IVAR) across valid B, R, and Z pixels."""
    if not coadd_path.is_file():
        return np.nan, 0, "missing_file"

    try:
        with fits.open(coadd_path, memmap=True) as hdul:
            row_idx = get_target_row_idx(hdul, target_id)
            if row_idx is None:
                return np.nan, 0, "target_not_in_fibermap"

            snr_chunks = []
            for band in ("B", "R", "Z"):
                flux_key = f"{band}_FLUX"
                ivar_key = f"{band}_IVAR"
                if flux_key not in hdul or ivar_key not in hdul:
                    continue

                flux = select_spectrum(hdul[flux_key].data, row_idx).astype(
                    np.float64, copy=False
                )
                ivar = select_spectrum(hdul[ivar_key].data, row_idx).astype(
                    np.float64, copy=False
                )
                valid = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
                if np.any(valid):
                    snr_chunks.append(flux[valid] * np.sqrt(ivar[valid]))

            if not snr_chunks:
                return np.nan, 0, "no_valid_pixels"

            all_snr = np.concatenate(snr_chunks)
            all_snr = all_snr[np.isfinite(all_snr)]
            if all_snr.size == 0:
                return np.nan, 0, "no_finite_snr"

            return float(np.median(all_snr)), int(all_snr.size), "ok"

    except Exception as error:
        print(
            f"Could not calculate S/N for {coadd_path}: "
            f"{type(error).__name__}: {error}"
        )
        return np.nan, 0, f"error:{type(error).__name__}"


def apply_snr_cut(
    df,
    coadd_root,
    snr_cut=SNR_CUT,
    min_kept=MIN_KEPT_COADDS,
):
    """Keep passing coadd rows and targets having at least ``min_kept`` rows."""
    evaluated_rows = []

    for _, row in df.iterrows():
        coadd_path = coadd_path_for_row(row, coadd_root)
        median_snr, n_snr_pixels, status = calculate_median_snr(
            coadd_path, row["TARGETID"]
        )

        evaluated = row.copy()
        evaluated["COADD_PATH"] = str(coadd_path)
        evaluated["MEDIAN_SNR"] = median_snr
        evaluated["SNR_NPIX"] = n_snr_pixels
        evaluated["SNR_STATUS"] = status
        evaluated["PASSES_SNR_FLAG"] = bool(
            np.isfinite(median_snr) and median_snr >= snr_cut
        )
        evaluated_rows.append(evaluated)

    evaluated_df = pd.DataFrame(evaluated_rows)
    if evaluated_df.empty:
        return evaluated_df, evaluated_df.copy()

    passing_df = evaluated_df.loc[evaluated_df["PASSES_SNR_FLAG"]].copy()
    # Count distinct coadd files, not merely catalog rows, so duplicated catalog
    # records cannot make a target pass the minimum-coadd requirement.
    passing_counts = passing_df.groupby("TARGETID")["COADD_PATH"].nunique()
    kept_ids = passing_counts.loc[passing_counts >= min_kept].index
    final_df = passing_df.loc[passing_df["TARGETID"].isin(kept_ids)].copy()
    return final_df, evaluated_df


def ensure_parent_directories(*paths):
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    if not args.coadd_root.is_dir():
        raise FileNotFoundError(f"Coadd root does not exist: {args.coadd_root}")

    df = load_catalog(args.fits_catalog)

    print("\nStage 0: Initial catalog")
    print(f"  Rows:      {len(df)}")
    print(f"  TARGETIDs: {df['TARGETID'].nunique()}")

    before = df
    df = apply_quality_flags(df)
    report_filter("Stage 1: Quality flags", before, df)

    before = df
    df = apply_redshift_window(df)
    report_filter(
        f"Stage 2: At least one observation satisfies {Z_MIN} <= Z <= {Z_MAX}",
        before,
        df,
    )

    before = df
    df = apply_repeat_observation_filter(df)
    report_filter("Stage 3: Repeat observations", before, df)

    before = df
    df = apply_duration_filter(df)
    report_filter(f"Stage 4: Duration >= {MIN_DURATION_DAYS} days", before, df)

    before = df
    df, diagnostics = apply_snr_cut(df, args.coadd_root)
    report_filter(
        f"Stages 5-6: S/N >= {SNR_CUT} and >= {MIN_KEPT_COADDS} passing coadds",
        before,
        df,
    )

    ensure_parent_directories(
        args.output_csv, args.target_output, args.snr_diagnostics
    )

    diagnostics.to_csv(args.snr_diagnostics, index=False)

    if not df.empty:
        df = df.sort_values(
            ["duration_days", "TARGETID", "LASTNIGHT"],
            ascending=[False, True, True],
        )
    df.to_csv(args.output_csv, index=False)

    unique_targets = (
        df[["TARGETID"]].drop_duplicates().sort_values("TARGETID")
        if not df.empty
        else pd.DataFrame(columns=["TARGETID"])
    )
    unique_targets.to_csv(args.target_output, index=False)

    status = diagnostics.get("SNR_STATUS", pd.Series(dtype=str))
    missing_count = int(status.eq("missing_file").sum())
    invalid_count = int((~status.isin(["ok", "missing_file"])).sum())

    print("\n========== FINAL SUMMARY ==========")
    print(f"Coadds evaluated:          {len(diagnostics)}")
    print(f"Missing coadd files:       {missing_count}")
    print(f"Other invalid coadds:      {invalid_count}")
    print(f"Final catalog rows:        {len(df)}")
    print(f"Final unique TARGETIDs:    {df['TARGETID'].nunique()}")
    print(f"Full output:               {args.output_csv}")
    print(f"Unique TARGETID output:    {args.target_output}")
    print(f"S/N diagnostics:           {args.snr_diagnostics}")
    print("===================================")


if __name__ == "__main__":
    main()
