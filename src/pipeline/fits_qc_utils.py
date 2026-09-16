"""
Shared helpers for applying per-coadd S/N quality cuts to DESI FITS files.

Used by both quality_cut.py and pipeline.py so the S/N classification logic
only needs to be maintained in one place.
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
    """Return the observing date as YYYY-MM-DD, or 'unknown'."""
    raw = get_obs_date_raw(filepath)
    if raw.isdigit() and len(raw) == 8:
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
    return "unknown"


def row_select(arr, row_idx):
    """Select one target row while preserving already one-dimensional arrays."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    return arr[row_idx]


def get_target_row_idx(hdul, target_id):
    """Find a TARGETID in FIBERMAP, defaulting to row zero when unavailable."""
    if "FIBERMAP" not in hdul:
        return 0

    fmap = hdul["FIBERMAP"].data
    if getattr(fmap, "names", None) is None or "TARGETID" not in fmap.names:
        return 0

    target_ids = np.asarray(fmap["TARGETID"])
    matching = np.where(target_ids.astype(str) == str(target_id))[0]
    return int(matching[0]) if matching.size > 0 else 0


def compute_median_snr_from_original_coadd(coadd_path, target_id):
    """Calculate median per-pixel S/N across the B, R, and Z arms."""
    if coadd_path is None or not os.path.exists(coadd_path):
        return np.nan, 0

    try:
        with fits.open(coadd_path) as hdul:
            row_idx = get_target_row_idx(hdul, target_id)
            snr_chunks = []

            for arm in ("B", "R", "Z"):
                flux_key = f"{arm}_FLUX"
                ivar_key = f"{arm}_IVAR"

                if flux_key not in hdul or ivar_key not in hdul:
                    continue

                flux = row_select(
                    hdul[flux_key].data,
                    row_idx,
                ).astype(np.float64)

                ivar = row_select(
                    hdul[ivar_key].data,
                    row_idx,
                ).astype(np.float64)

                valid = (
                    np.isfinite(flux)
                    & np.isfinite(ivar)
                    & (ivar > 0)
                )

                if not np.any(valid):
                    continue

                snr = flux[valid] * np.sqrt(ivar[valid])
                snr = snr[np.isfinite(snr)]

                if snr.size > 0:
                    snr_chunks.append(snr)

            if not snr_chunks:
                return np.nan, 0

            all_snr = np.concatenate(snr_chunks)
            return float(np.median(all_snr)), int(all_snr.size)

    except Exception as error:
        print(
            f"Could not calculate S/N for {coadd_path}: "
            f"{type(error).__name__}: {error}"
        )
        return np.nan, 0


def classify_target_by_snr(target_id, coadd_root, plot_root, snr_cut):
    """
    Classify every original coadd for one target using median per-pixel S/N.

    A classification CSV is written under ``plot_root/<target_id>/``.
    """
    target_root = os.path.join(coadd_root, str(target_id))

    empty_result = {
        "target_id": str(target_id),
        "exists": False,
        "kept_files": [],
        "rejected_low": 0,
        "invalid": 0,
        "total_coadds": 0,
    }

    if not os.path.isdir(target_root):
        return empty_result

    coadd_files = sorted(
        glob.glob(os.path.join(target_root, "coadd-*.fits")),
        key=get_obs_date_raw,
    )

    if not coadd_files:
        return {
            **empty_result,
            "exists": True,
        }

    out_dir = os.path.join(plot_root, str(target_id))
    os.makedirs(out_dir, exist_ok=True)

    kept_files = []
    rejected_low_count = 0
    invalid_count = 0
    table_rows = []

    for coadd_path in coadd_files:
        median_snr, n_snr_pix = compute_median_snr_from_original_coadd(
            coadd_path,
            target_id,
        )

        is_valid = np.isfinite(median_snr)
        passes_snr = is_valid and median_snr >= snr_cut
        rejected_low = is_valid and median_snr < snr_cut

        if passes_snr:
            kept_files.append((coadd_path, median_snr))
        elif rejected_low:
            rejected_low_count += 1
        else:
            invalid_count += 1

        table_rows.append({
            "TARGETID": str(target_id),
            "OBS_DATE": get_obs_date_label(coadd_path),
            "FILE_COADD": os.path.basename(coadd_path),
            "MEDIAN_SNR": median_snr,
            "SNR_NPIX": n_snr_pix,
            "PASSES_SNR_FLAG": bool(passes_snr),
            "KEPT_FLAG": bool(passes_snr),
            "REJECTED_LOW_SNR_FLAG": bool(rejected_low),
            "INVALID_SNR_FLAG": not is_valid,
        })

    table = pd.DataFrame(table_rows).sort_values("OBS_DATE")
    table.to_csv(
        os.path.join(out_dir, f"{target_id}_median_snr_by_coadd.csv"),
        index=False,
    )

    return {
        "target_id": str(target_id),
        "exists": True,
        "kept_files": kept_files,
        "rejected_low": rejected_low_count,
        "invalid": invalid_count,
        "total_coadds": len(coadd_files),
    }


def screen_targets_by_snr(
    coadd_root,
    plot_root,
    allowed_target_ids=None,
    snr_cut=2.0,
    min_kept=2,
):
    """
    Screen target directories using only median per-coadd S/N.

    Targets are retained when at least ``min_kept`` original coadds have a
    finite median S/N greater than or equal to ``snr_cut``.

    Returns
    -------
    snr_ok_map : dict
        Maps each retained TARGETID to its classification summary.
    stage_stats : dict
        Aggregate counts for the screening stage.
    """
    if not os.path.isdir(coadd_root):
        raise FileNotFoundError(f"Coadd root does not exist: {coadd_root}")

    target_dirs = sorted(
        directory
        for directory in os.listdir(coadd_root)
        if os.path.isdir(os.path.join(coadd_root, directory))
    )

    if allowed_target_ids is not None:
        allowed = {str(target_id) for target_id in allowed_target_ids}
        target_dirs = [
            directory
            for directory in target_dirs
            if str(directory) in allowed
        ]

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

        summary = classify_target_by_snr(
            target_id=target_id,
            coadd_root=coadd_root,
            plot_root=plot_root,
            snr_cut=snr_cut,
        )

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
