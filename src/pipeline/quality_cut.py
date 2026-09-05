#!/usr/bin/env python
# coding: utf-8

from astropy.io import fits
import numpy as np
import pandas as pd
from pathlib import Path


"""

Assuming the current directory is 
/work/11161/kanyuni/ls6/quassiQ_project/quassiQ/src/pipeline/quality_cut.py

~/ls6/quassiQ_project (your directory for the project)
|        |───quassiQ (git clone)
|            └── src/pipeline/quality_cut.py(this script)
└────────────coadds/

"""

quassiQ_root = Path(__file__).resolve().parents[2] #3 up so /quassiQ/
coadd_root = quassiQ_root / 'coadds'


PROJECT_ROOT = THIS_FILE.parents[3]   # -> .../quassiQ_project

# You likely need to chnage this path below
fits_file = PROJECT_ROOT / "QSO_iron" / "iron" / "QSO_cat_iron_cumulative_v0.fits"

# Load FITS table
with fits.open(fits_file) as hdul:
    # prevents potential FITS byte-order/endianness issues in pandas
    data = np.ascontiguousarray(hdul[1].data)

df = pd.DataFrame(data)
print(f"Initial row count: {len(df)}")

# ===== helper =====

# snr helper
def compute_median_snr_from_original_coadd(coadd_path, target_id):
    if coadd_path is None or not os.path.exists(coadd_path):
        return np.nan, 0

    try:
        with fits.open(coadd_path) as hdul:
            row_idx = get_target_row_idx(hdul, target_id)
            snr_chunks = []

            for arm in ["B", "R", "Z"]:
                flux_key = f"{arm}_FLUX"
                ivar_key = f"{arm}_IVAR"
                if flux_key not in hdul or ivar_key not in hdul:
                    continue

                flux = row_select(hdul[flux_key].data, row_idx).astype(np.float64)
                ivar = row_select(hdul[ivar_key].data, row_idx).astype(np.float64)

                good = np.isfinite(flux) & np.isfinite(ivar) & (ivar > 0)
                if np.any(good):
                    snr = flux[good] * np.sqrt(ivar[good])
                    snr = snr[np.isfinite(snr)]
                    if snr.size > 0:
                        snr_chunks.append(snr)

            if not snr_chunks:
                return np.nan, 0

            all_snr = np.concatenate(snr_chunks)
            return float(np.median(all_snr)), int(all_snr.size)
    except Exception:
        return np.nan, 0


def compute_reduced_chi2_from_weight_hdu(recon_path, target_id):
    if recon_path is None or not os.path.exists(recon_path):
        return np.nan

    try:
        with fits.open(recon_path) as hdul:
            if "OBSERVED" not in hdul or "RECON" not in hdul or "WEIGHT" not in hdul:
                return np.nan

            row_idx = get_target_row_idx(hdul, target_id)

            obs = hdul["OBSERVED"].data
            rec = hdul["RECON"].data
            weight = hdul["WEIGHT"].data

            wave_obs = row_select(obs["WAVE_REST"], row_idx).astype(np.float64)
            flux_obs = row_select(obs["OBS_FLUX"], row_idx).astype(np.float64)
            wave_rec = row_select(rec["WAVE_RECON"], row_idx).astype(np.float64)
            flux_rec = row_select(rec["RECON_FLUX"], row_idx).astype(np.float64)
            weights = row_select(weight["W_UPDATED"], row_idx).astype(np.float64)

            good = np.isfinite(wave_obs) & np.isfinite(flux_obs) & np.isfinite(weights) & (weights > 0)
            if not np.any(good):
                return np.nan

            recon_interp = np.interp(wave_obs, wave_rec, flux_rec, left=np.nan, right=np.nan)
            valid = good & np.isfinite(recon_interp)
            if not np.any(valid):
                return np.nan

            chi2_values = (flux_obs[valid] - recon_interp[valid]) ** 2 * weights[valid]
            return float(np.sum(chi2_values) / valid.sum())
    except Exception:
        return np.nan


def classify_target_by_snr(target_id):
    target_root = os.path.join(coadd_root, str(target_id))
    recon_dir = os.path.join(target_root, "recon")
    out_dir = os.path.join(plot_root, str(target_id))
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(target_root):
        return {
            "target_id": str(target_id),
            "exists": False,
            "kept_files": [],
            "rejected_low": 0,
            "invalid": 0,
            "total_recon": 0,
        }
    if not os.path.isdir(recon_dir):
        return {
            "target_id": str(target_id),
            "exists": False,
            "kept_files": [],
            "rejected_low": 0,
            "invalid": 0,
            "total_recon": 0,
        }

    recon_files = sorted(glob.glob(os.path.join(recon_dir, "*_recon.fits")), key=get_obs_date_raw)
    if not recon_files:
        recon_files = sorted(glob.glob(os.path.join(recon_dir, "*.fits")), key=get_obs_date_raw)

    coadd_files = sorted(glob.glob(os.path.join(target_root, "coadd-*.fits")), key=get_obs_date_raw)
    coadd_by_date = {get_obs_date_raw(path): path for path in coadd_files}

    kept_files = []
    rejected_low_snr_count = 0
    rejected_high_chi2_count = 0
    invalid_snr_count = 0
    invalid_chi2_count = 0
    snr_table_rows = []

    for recon_path in recon_files:
        date_key = get_obs_date_raw(recon_path)
        coadd_path = coadd_by_date.get(date_key)

        median_snr, n_snr_pix = compute_median_snr_from_original_coadd(coadd_path, target_id)

        passes_snr = np.isfinite(median_snr) and (median_snr >= snr_cut)
        rejected_low_snr = np.isfinite(median_snr) and (median_snr < snr_cut)
        invalid_snr = not np.isfinite(median_snr)

        chi2_weight = np.nan
        passes_chi2_weight = False
        rejected_high_chi2 = False
        invalid_chi2 = False

        if passes_snr:
            chi2_weight = compute_reduced_chi2_from_weight_hdu(recon_path, target_id)
            passes_chi2_weight = np.isfinite(chi2_weight) and (chi2_weight <= chi2_cut)
            rejected_high_chi2 = np.isfinite(chi2_weight) and (chi2_weight > chi2_cut)
            invalid_chi2 = not np.isfinite(chi2_weight)

        is_kept = bool(passes_snr and passes_chi2_weight)

        if is_kept:
            kept_files.append((recon_path, median_snr))
        elif rejected_low_snr:
            rejected_low_snr_count += 1
        elif rejected_high_chi2:
            rejected_high_chi2_count += 1
        elif invalid_chi2:
            invalid_chi2_count += 1
        else:
            invalid_snr_count += 1

        snr_table_rows.append(
            {
                "TARGETID": str(target_id),
                "OBS_DATE": get_obs_date_label(recon_path),
                "FILE_RECON": os.path.basename(recon_path),
                "FILE_COADD": os.path.basename(coadd_path) if coadd_path else "",
                "MEDIAN_SNR": median_snr,
                "SNR_NPIX": n_snr_pix,
                "CHI2_WEIGHT": chi2_weight,
                "PASSES_SNR_FLAG": bool(passes_snr),
                "PASSES_CHI2_WEIGHT_FLAG": bool(passes_chi2_weight),
                "KEPT_FLAG": bool(is_kept),
                "REJECTED_LOW_SNR_FLAG": bool(rejected_low_snr),
                "REJECTED_HIGH_CHI2_FLAG": bool(rejected_high_chi2),
                "INVALID_SNR_FLAG": bool(invalid_snr),
                "INVALID_CHI2_FLAG": bool(invalid_chi2),
            }
        )

    if snr_table_rows:
        snr_df = pd.DataFrame(snr_table_rows).sort_values("OBS_DATE")
        snr_csv = os.path.join(out_dir, f"{target_id}_median_snr_by_coadd.csv")
        snr_df.to_csv(snr_csv, index=False)

    return {
        "target_id": str(target_id),
        "exists": True,
        "kept_files": kept_files,
        "rejected_low": rejected_low_snr_count,
        "rejected_high_chi2": rejected_high_chi2_count,
        "invalid": invalid_snr_count,
        "invalid_chi2": invalid_chi2_count,
        "total_recon": len(recon_files),
    }


def screen_targets_by_snr(allowed_target_ids=None, min_kept=2):
    target_dirs = sorted(
        d for d in os.listdir(coadd_root)
        if os.path.isdir(os.path.join(coadd_root, d))
    )

    if allowed_target_ids is not None:
        target_dirs = [d for d in target_dirs if str(d) in allowed_target_ids]

    snr_ok_map = {}
    stage_stats = {
        "targets_scanned": 0,
        "targets_with_min_kept": 0,
        "targets_removed_lt_min_kept": 0,
        "total_kept_coadds": 0,
        "total_rejected_low": 0,
        "total_invalid": 0,
    }

    for target_id in target_dirs:
        stage_stats["targets_scanned"] += 1
        summary = classify_target_by_snr(target_id)

        kept_n = len(summary["kept_files"])
        stage_stats["total_kept_coadds"] += kept_n
        stage_stats["total_rejected_low"] += summary["rejected_low"]
        stage_stats["total_invalid"] += summary["invalid"]

        if kept_n >= min_kept:
            snr_ok_map[str(target_id)] = summary
            stage_stats["targets_with_min_kept"] += 1
        else:
            stage_stats["targets_removed_lt_min_kept"] += 1

    return snr_ok_map, stage_stats


# 1. Quality Filtering (ZWARN & COADD_FIBERSTATUS)
df = df[df['ZWARN'].isin([0, 4])]
df = df[df['COADD_FIBERSTATUS'] == 0]
print(f"Rows after quality filtering: {len(df)}")


# 2. Redshift Filter (2.1 <= Z <= 3.5)
mask_z = (df['Z'] >= 2.1) & (df['Z'] <= 3.5)
valid_z_ids = df.loc[mask_z, 'TARGETID'].unique()
df = df[df['TARGETID'].isin(valid_z_ids)]
print(f"Rows after redshift filtering: {len(df)}")


# 3. Filter for Duplicate Observations, keep TARGETIDs that have more than one valid observation remaining
df = df[df['TARGETID'].duplicated(keep=False)]
print(f"Rows with multiple observations: {len(df)}")

# Datetime Conversion & Duration Calculation
df['LASTNIGHT'] = pd.to_datetime(df['LASTNIGHT'], format='%Y%m%d')

duration_df = df.groupby('TARGETID')['LASTNIGHT'].agg(['min', 'max'])
duration_df['duration_days'] = (duration_df['max'] - duration_df['min']).dt.days

# Merge duration back to main DataFrame
df = df.merge(duration_df['duration_days'], on='TARGETID')

# Filter by Minimum Duration (>= 30 days)
valid_duration_ids = df.loc[df['duration_days'] >= 30, 'TARGETID'].unique()
df = df[df['TARGETID'].isin(valid_duration_ids)]

# 4. S/R cut

snr_ok_map, snr_stats = screen_targets_by_snr(
    allowed_target_ids=redshift_eligible_ids,
    min_kept=min_kept_coadds_per_target,
)
snr_ok_targets = set(snr_ok_map.keys())


# 5. chi2

kept_files = summary["kept_files"]

    if len(kept_files) < min_kept_coadds_per_target:
        continue

    plot_result = plot_target_from_kept_files(target_id, kept_files)
    passed_flux_ratio = plot_result["passed_flux_ratio"]
    peak_summary_df = plot_result["peak_summary_df"]

    if not peak_summary_df.empty:
        peak_summary_frames.append(peak_summary_df)

    flux_ratio_inventory_rows.append(
        {
            "TARGETID": str(target_id),
            "KEPT_COADDS": len(kept_files),
            "REJECTED_LOW_SNR": summary["rejected_low"],
            "REJECTED_HIGH_CHI2": summary.get("rejected_high_chi2", 0),
            "INVALID_SNR": summary["invalid"],
            "INVALID_CHI2": summary.get("invalid_chi2", 0),
            "TOTAL_RECON": summary["total_recon"],
            "PASSED_FLUX_RATIO_CUT": bool(passed_flux_ratio),
        }
    )

    if not passed_flux_ratio:
        total_flux_ratio_discarded += 1
        print(
            f"Discarding {target_id}: kept={len(kept_files)}, "
            f"low={summary['rejected_low']}, chi2={summary.get('rejected_high_chi2', 0)}, "
            f"invalid={summary['invalid']}, total={summary['total_recon']}"
        )
        continue

# Final Sort & Export
df = df.sort_values(['duration_days', 'TARGETID'], ascending=[False, True])

output_csv = "CLQ_candidates.csv"
df.to_csv(output_csv, index=False)

print("\n Summary")
print(f"Saved final candidates to: {output_csv}")
print(f"Final remaining rows: {len(df)}")
print(f"Final unique TARGETIDs: {df['TARGETID'].nunique()}")