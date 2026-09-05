"""
Shared helpers for working with DESI coadd / reconstructed-spectra FITS files.

Used by both quality_cut.py and pipeline.py so the S/N + chi2 classification
logic only needs to be maintained in one place.
"""

import glob
import os

import numpy as np
import pandas as pd
from astropy.io import fits


def get_obs_date_raw(filepath):
    """Extract an 8-digit YYYYMMDD token from a filename, else 'unknown'."""
    name = os.path.basename(filepath)
    for part in name.split("-"):
        if len(part) == 8 and part.isdigit():
            return part
    return "unknown"


def get_obs_date_label(filepath):
    raw = get_obs_date_raw(filepath)
    if raw.isdigit() and len(raw) == 8:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return "unknown"


def row_select(arr, row_idx):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    return arr[row_idx]


def get_target_row_idx(hdul, target_id):
    if "FIBERMAP" not in hdul:
        return 0
    fmap = hdul["FIBERMAP"].data
    if getattr(fmap, "names", None) is None or "TARGETID" not in fmap.names:
        return 0
    tids = np.asarray(fmap["TARGETID"])
    idx = np.where(tids.astype(str) == str(target_id))[0]
    return int(idx[0]) if idx.size > 0 else 0


def compute_median_snr_from_original_coadd(coadd_path, target_id):
    """Median per-pixel S/N (flux * sqrt(ivar)) across the B/R/Z arms."""
    if coadd_path is None or not os.path.exists(coadd_path):
        return np.nan, 0
    try:
        with fits.open(coadd_path) as hdul:
            row_idx = get_target_row_idx(hdul, target_id)
            snr_chunks = []
            for arm in ["B", "R", "Z"]:
                flux_key, ivar_key = f"{arm}_FLUX", f"{arm}_IVAR"
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
    """Reduced chi2 between observed and reconstructed flux, weighted by W_UPDATED."""
    if recon_path is None or not os.path.exists(recon_path):
        return np.nan
    try:
        with fits.open(recon_path) as hdul:
            if "OBSERVED" not in hdul or "RECON" not in hdul or "WEIGHT" not in hdul:
                return np.nan
            row_idx = get_target_row_idx(hdul, target_id)
            obs, rec, weight = hdul["OBSERVED"].data, hdul["RECON"].data, hdul["WEIGHT"].data

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


def classify_target_by_snr(target_id, coadd_root, plot_root, snr_cut, chi2_cut):
    """
    For one target: walk its reconstructed-spectra files, compute median S/N
    (from the matching original coadd) and reduced chi2 (from the recon
    file's weight HDU), and classify each observation epoch as
    kept / rejected-low-S/N / rejected-high-chi2 / invalid.

    Writes a per-target CSV of the classification under
    `plot_root/<target_id>/`.
    """
    target_root = os.path.join(coadd_root, str(target_id))
    recon_dir = os.path.join(target_root, "recon")

    empty_result = {
        "target_id": str(target_id),
        "exists": False,
        "kept_files": [],
        "rejected_low": 0,
        "rejected_high_chi2": 0,
        "invalid": 0,
        "invalid_chi2": 0,
        "total_recon": 0,
    }

    if not os.path.isdir(target_root) or not os.path.isdir(recon_dir):
        return empty_result

    out_dir = os.path.join(plot_root, str(target_id))
    os.makedirs(out_dir, exist_ok=True)

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

        snr_table_rows.append({
            "TARGETID": str(target_id),
            "OBS_DATE": get_obs_date_label(recon_path),
            "FILE_RECON": os.path.basename(recon_path),
            "FILE_COADD": os.path.basename(coadd_path) if coadd_path else "",
            "MEDIAN_SNR": median_snr,
            "SNR_NPIX": n_snr_pix,
            "CHI2_WEIGHT": chi2_weight,
            "PASSES_SNR_FLAG": bool(passes_snr),
            "PASSES_CHI2_WEIGHT_FLAG": bool(passes_chi2_weight),
            "KEPT_FLAG": is_kept,
            "REJECTED_LOW_SNR_FLAG": bool(rejected_low_snr),
            "REJECTED_HIGH_CHI2_FLAG": bool(rejected_high_chi2),
            "INVALID_SNR_FLAG": not np.isfinite(median_snr),
            "INVALID_CHI2_FLAG": bool(invalid_chi2),
        })

    if snr_table_rows:
        snr_df = pd.DataFrame(snr_table_rows).sort_values("OBS_DATE")
        snr_df.to_csv(os.path.join(out_dir, f"{target_id}_median_snr_by_coadd.csv"), index=False)

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


def screen_targets_by_snr(coadd_root, plot_root, allowed_target_ids=None,
                           snr_cut=2.0, chi2_cut=10.0, min_kept=2):
    """
    Classify every target directory under `coadd_root` (optionally restricted
    to `allowed_target_ids`) and keep those with >= min_kept coadds passing
    both the S/N and chi2 cuts.

    Returns (snr_ok_map, stage_stats), where snr_ok_map maps
    str(target_id) -> classification summary dict (see classify_target_by_snr).
    """
    target_dirs = sorted(
        d for d in os.listdir(coadd_root) if os.path.isdir(os.path.join(coadd_root, d))
    )

    if allowed_target_ids is not None:
        allowed = {str(t) for t in allowed_target_ids}
        target_dirs = [d for d in target_dirs if str(d) in allowed]

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
        summary = classify_target_by_snr(target_id, coadd_root, plot_root, snr_cut, chi2_cut)

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