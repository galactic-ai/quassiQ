import os
import glob
import math
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits


# ============================================================
# COMBINED PIPELINE (FINAL ORDER)
# 0) Redshift filter first (Z > 2.4)
# 1) S/N filter: keep coadds with median S/N >= snr_cut
# 2) Drop targets with fewer than min_kept_coadds_per_target kept coadds
# 3) Latent filter: keep targets with n_latents_exceed_p95 > 0
# 4) Plot final intersection targets
# ============================================================


# -------------------------
# PATHS / CONFIG
# -------------------------
latent_csv = "/work2/11161/kanyuni/ls6/quassiQ/latent/latent_all_targets_13927.csv"
catalog_csv = "/work2/11161/kanyuni/ls6/quassiQ/catalog/CLQ_candidates.csv"
latent_out_dir = "/work2/11161/kanyuni/ls6/quassiQ/latent"

coadd_root = "/work2/11161/kanyuni/ls6/quassiQ/coadds"
plot_root = "/work2/11161/kanyuni/ls6/quassiQ/plot/0421"
flux_ratio_pass_root = os.path.join(plot_root, "flux_ratio_pass")

snr_cut = 2.0
min_kept_coadds_per_target = 2
z_min = 2.4
flux_ratio_cut = 0.5
flux_ratio_lines = ("LyA", "C IV", "C III]")

bin_points = 30
fixed_xlim = (1125, 2125)
smooth_width = 1
latent_base_cols = [f"Latent{i}" for i in range(1, 11)]

line_cfg = {
    "O VI": {"center": 1033.82, "window": 20.0, "color": "tab:gray"},
    "LyA": {"center": 1215.0, "window": 25.0, "color": "tab:gray"},
    "N V": {"center": 1240.0, "window": 25.0, "color": "tab:gray"},
    "C IV": {"center": 1549.48, "window": 35.0, "color": "tab:gray"},
    "C III]": {"center": 1908.734, "window": 45.0, "color": "tab:gray"},
    "Mg II": {"center": 2799.117, "window": 55.0, "color": "tab:gray"},
}


def make_line_slug(line_name):
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(line_name)).strip("_")


def get_obs_date_raw(filepath):
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


def coarse_bin(x, y, n_points=30):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size == 0:
        return x, y

    n_points = max(1, int(n_points))
    n = x.size // n_points
    if n == 0:
        return x, y

    trim = n * n_points
    xb = np.nanmean(x[:trim].reshape(n, n_points), axis=1)
    yb = np.nanmean(y[:trim].reshape(n, n_points), axis=1)
    return xb, yb


def finite_minmax(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(values.min()), float(values.max())


def mask_to_range(x, lo, hi):
    x = np.asarray(x, dtype=np.float64)
    return np.isfinite(x) & (x >= lo) & (x <= hi)


def recon_ylim(*arrays):
    values = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size > 0:
            values.append(arr)
    if not values:
        return None

    vals = np.concatenate(values)
    ymin = float(np.min(vals))
    ymax = float(np.max(vals))
    if ymax <= ymin:
        return ymin - 1.0, ymax + 1.0
    return ymin, ymax


def smooth_flux(y, width=9):
    y = np.asarray(y, dtype=np.float64)
    if width <= 1:
        return y
    width = int(width)
    if width % 2 == 0:
        width += 1
    if y.size < width:
        return y
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(y, kernel, mode="same")


def find_local_peak_in_window(wave, flux, center, half_window):
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    good = np.isfinite(wave) & np.isfinite(flux)
    wave = wave[good]
    flux = flux[good]
    if wave.size == 0:
        return np.nan, np.nan

    sel = (wave >= center - half_window) & (wave <= center + half_window)
    if not np.any(sel):
        return np.nan, np.nan

    w_sub = wave[sel]
    f_sub = flux[sel]
    if w_sub.size == 0:
        return np.nan, np.nan

    j = int(np.nanargmax(f_sub))
    return float(w_sub[j]), float(f_sub[j])


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


def load_redshift_eligible_target_ids():
    catalog_df = pd.read_csv(catalog_csv)
    if "TARGETID" not in catalog_df.columns or "Z" not in catalog_df.columns:
        raise ValueError("Catalog must contain TARGETID and Z columns.")

    catalog_df = catalog_df[["TARGETID", "Z"]].copy()
    catalog_df["TARGETID"] = pd.to_numeric(catalog_df["TARGETID"], errors="coerce").astype("Int64")
    catalog_df["Z"] = pd.to_numeric(catalog_df["Z"], errors="coerce")

    eligible = set(
        catalog_df.loc[catalog_df["Z"] > z_min, "TARGETID"].dropna().astype(int).astype(str).unique()
    )

    return eligible, len(catalog_df)


@lru_cache(maxsize=1)
def load_catalog_redshift_map():
    catalog_df = pd.read_csv(catalog_csv, usecols=["TARGETID", "Z"])
    catalog_df["TARGETID"] = pd.to_numeric(catalog_df["TARGETID"], errors="coerce").astype("Int64")
    catalog_df["Z"] = pd.to_numeric(catalog_df["Z"], errors="coerce")
    catalog_df = catalog_df.dropna(subset=["TARGETID", "Z"]).copy()
    catalog_df["TARGETID"] = catalog_df["TARGETID"].astype(int).astype(str)
    return dict(zip(catalog_df["TARGETID"], catalog_df["Z"]))


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
    invalid_snr_count = 0
    snr_table_rows = []

    for recon_path in recon_files:
        date_key = get_obs_date_raw(recon_path)
        coadd_path = coadd_by_date.get(date_key)

        median_snr, n_snr_pix = compute_median_snr_from_original_coadd(coadd_path, target_id)

        is_kept = np.isfinite(median_snr) and (median_snr >= snr_cut)
        is_rejected_low = np.isfinite(median_snr) and (median_snr < snr_cut)
        is_invalid = not np.isfinite(median_snr)

        if is_kept:
            kept_files.append((recon_path, median_snr))
        elif is_rejected_low:
            rejected_low_snr_count += 1
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
                "KEPT_FLAG": bool(is_kept),
                "REJECTED_LOW_SNR_FLAG": bool(is_rejected_low),
                "INVALID_SNR_FLAG": bool(is_invalid),
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
        "invalid": invalid_snr_count,
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


def compute_target_latent_counts(allowed_target_ids=None):
    df = pd.read_csv(latent_csv)

    mu_df = (
        df.groupby("TARGETID", as_index=False)[latent_base_cols]
        .mean()
        .rename(columns={column: f"{column}_mu" for column in latent_base_cols})
    )

    std_df = (
        df.groupby("TARGETID", as_index=False)[latent_base_cols]
        .std(ddof=0)
        .rename(columns={column: f"{column}_std" for column in latent_base_cols})
    )

    out_df = std_df.merge(mu_df, on="TARGETID", how="left")
    out_df["TARGETID"] = pd.to_numeric(out_df["TARGETID"], errors="coerce").astype("Int64")

    if allowed_target_ids is not None:
        allowed_int = set(int(tid) for tid in allowed_target_ids)
        out_df = out_df[out_df["TARGETID"].isin(allowed_int)].copy()

    norm_cols = []
    for column in latent_base_cols:
        mu_col = f"{column}_mu"
        std_col = f"{column}_std"
        norm_col = f"{column}_std_normalized"
        out_df[norm_col] = out_df[std_col] / out_df[mu_col].abs().replace(0, np.nan)
        norm_cols.append(norm_col)

    p95_map = {}
    for column in norm_cols:
        values = pd.to_numeric(out_df[column], errors="coerce").abs().dropna()
        p95_map[column] = float(values.quantile(0.95)) if values.size > 0 else np.nan

    for column in norm_cols:
        thr = p95_map[column]
        out_df[f"{column}_above_p95"] = out_df[column].abs() > thr

    flag_cols = [f"{column}_above_p95" for column in norm_cols]
    out_df["n_latents_exceed_p95"] = out_df[flag_cols].sum(axis=1)
    out_df = out_df.sort_values("n_latents_exceed_p95", ascending=False).reset_index(drop=True)
    out_df["TARGETID"] = out_df["TARGETID"].astype("Int64").astype(str)

    counts_csv = os.path.join(latent_out_dir, "latent_std_normalized_p95_counts_by_target.csv")
    out_df[["TARGETID", "n_latents_exceed_p95"] + norm_cols].to_csv(counts_csv, index=False)

    p95_csv = os.path.join(latent_out_dir, "latent_std_normalized_p95_by_latent.csv")
    p95_df = pd.DataFrame(
        {
            "latent_std_normalized_col": norm_cols,
            "p95_abs_std_normalized": [p95_map[column] for column in norm_cols],
        }
    )
    p95_df.to_csv(p95_csv, index=False)

    selected_targets_df = out_df.loc[
        out_df["n_latents_exceed_p95"] > 0,
        ["TARGETID", "n_latents_exceed_p95"],
    ].copy()
    selected_csv = os.path.join(latent_out_dir, "latent_targets_with_nonzero_p95_counts.csv")
    selected_targets_df.to_csv(selected_csv, index=False)

    print(f"Saved: {counts_csv}")
    print(f"Saved: {p95_csv}")
    print(f"Saved: {selected_csv}")

    return out_df, selected_targets_df


def summarize_peak_flux_changes(target_id, peak_df):
    required_cols = {"LINE", "OBS_DATE", "PEAK_FLUX"}
    if peak_df.empty or not required_cols.issubset(peak_df.columns):
        return pd.DataFrame()

    summary_rows = []
    clean_df = peak_df.copy()
    clean_df["PEAK_FLUX"] = pd.to_numeric(clean_df["PEAK_FLUX"], errors="coerce")
    clean_df = clean_df.dropna(subset=["LINE", "OBS_DATE", "PEAK_FLUX"])

    for line_name in sorted(clean_df["LINE"].unique()):
        line_df = clean_df[clean_df["LINE"] == line_name].copy()
        if line_df.empty:
            continue

        idx_low = line_df["PEAK_FLUX"].idxmin()
        idx_high = line_df["PEAK_FLUX"].idxmax()
        row_low = line_df.loc[idx_low]
        row_high = line_df.loc[idx_high]

        flux_low = float(row_low["PEAK_FLUX"])
        flux_high = float(row_high["PEAK_FLUX"])
        flux_diff = flux_high - flux_low

        if np.isclose(flux_low, 0.0):
            flux_diff_over_lowest = np.nan
        else:
            flux_diff_over_lowest = flux_diff / abs(flux_low)

        summary_rows.append(
            {
                "TARGETID": str(target_id),
                "LINE": str(line_name),
                "LOWEST_NIGHT": str(row_low["OBS_DATE"]),
                "LOWEST_FLUX": flux_low,
                "HIGHEST_NIGHT": str(row_high["OBS_DATE"]),
                "HIGHEST_FLUX": flux_high,
                "FLUX_DIFF": flux_diff,
                "FLUX_DIFF_OVER_LOWEST": flux_diff_over_lowest,
            }
        )

    return pd.DataFrame(summary_rows)


def passes_flux_ratio_cut(summary_df):
    if summary_df.empty:
        return False

    candidate_df = summary_df[summary_df["LINE"].isin(flux_ratio_lines)].copy()
    if candidate_df.empty:
        return False

    ratios = pd.to_numeric(candidate_df["FLUX_DIFF_OVER_LOWEST"], errors="coerce")
    return bool((ratios > flux_ratio_cut).fillna(False).any())


def load_recon_obs_arrays(path):
    with fits.open(path) as hdul:
        if "OBSERVED" not in hdul or "RECON" not in hdul:
            return None

        obs = hdul["OBSERVED"].data
        rec = hdul["RECON"].data

        wave_obs = np.asarray(obs["WAVE_REST"], dtype=np.float64)
        flux_obs = np.asarray(obs["OBS_FLUX"], dtype=np.float64)
        wave_rec = np.asarray(rec["WAVE_RECON"], dtype=np.float64)
        flux_rec = np.asarray(rec["RECON_FLUX"], dtype=np.float64)

    good_obs = np.isfinite(wave_obs) & np.isfinite(flux_obs)
    good_rec = np.isfinite(wave_rec) & np.isfinite(flux_rec)
    wave_obs, flux_obs = wave_obs[good_obs], flux_obs[good_obs]
    wave_rec, flux_rec = wave_rec[good_rec], flux_rec[good_rec]

    if wave_rec.size == 0:
        return None

    if wave_obs.size > 0:
        obs_lohi = finite_minmax(wave_obs)
        if obs_lohi is not None:
            obs_lo, obs_hi = obs_lohi
            keep_rec = mask_to_range(wave_rec, obs_lo, obs_hi)
            wave_rec_plot = wave_rec[keep_rec]
            flux_rec_plot = flux_rec[keep_rec]
        else:
            wave_rec_plot, flux_rec_plot = wave_rec, flux_rec
    else:
        wave_rec_plot, flux_rec_plot = wave_rec, flux_rec

    return {
        "wave_obs": wave_obs,
        "flux_obs": flux_obs,
        "wave_rec": wave_rec_plot,
        "flux_rec": flux_rec_plot,
    }


def save_low_high_line_plots(target_id, out_dir, kept_file_records, peak_summary_df, redshift_label):
    if peak_summary_df.empty:
        return

    candidate_df = peak_summary_df[peak_summary_df["LINE"].isin(flux_ratio_lines)].copy()
    if candidate_df.empty:
        return

    file_record_by_date = {record["date_label"]: record for record in kept_file_records}

    for row in candidate_df.itertuples(index=False):
        line_name = str(row.LINE)
        cfg = line_cfg.get(line_name)
        if cfg is None:
            continue

        low_record = file_record_by_date.get(str(row.LOWEST_NIGHT))
        high_record = file_record_by_date.get(str(row.HIGHEST_NIGHT))
        if low_record is None or high_record is None:
            continue

        low_data = load_recon_obs_arrays(low_record["path"])
        high_data = load_recon_obs_arrays(high_record["path"])
        if low_data is None or high_data is None:
            continue

        window = cfg["window"] * 2.5
        x_lo = cfg["center"] - window
        x_hi = cfg["center"] + window

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
        panel_specs = [
            (axes[0], "Lowest peak flux", low_record, low_data, float(row.LOWEST_FLUX)),
            (axes[1], "Highest peak flux", high_record, high_data, float(row.HIGHEST_FLUX)),
        ]

        line_flux_ranges = []
        for ax, panel_title, record, data, peak_flux in panel_specs:
            wave_obs = data["wave_obs"]
            flux_obs = data["flux_obs"]
            wave_rec = data["wave_rec"]
            flux_rec = data["flux_rec"]

            if wave_obs.size > 0:
                obs_sel = mask_to_range(wave_obs, x_lo, x_hi)
                if np.any(obs_sel):
                    ax.plot(wave_obs[obs_sel], flux_obs[obs_sel], color="gray", alpha=0.12, lw=0.5, label="Obs raw")
                    wave_obs_c, flux_obs_c = coarse_bin(wave_obs[obs_sel], flux_obs[obs_sel], n_points=bin_points)
                    ax.plot(wave_obs_c, flux_obs_c, color="tab:blue", alpha=0.75, lw=2.0, linestyle="--", label="Obs coarse")

            rec_sel = mask_to_range(wave_rec, x_lo, x_hi)
            if not np.any(rec_sel):
                ax.axis("off")
                continue

            wave_rec_line = wave_rec[rec_sel]
            flux_rec_line = flux_rec[rec_sel]
            line_flux_ranges.append(flux_rec_line)

            ax.plot(wave_rec_line, flux_rec_line, color="tab:red", alpha=0.75, lw=2.2, label="Recon")
            ax.axvline(cfg["center"], color=cfg["color"], linestyle=":", lw=1.2, alpha=0.7)
            ax.set_xlim(x_lo, x_hi)
            ax.set_xlabel("Rest Wavelength (A)", fontsize=11)
            ax.set_ylabel("Flux", fontsize=11)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(alpha=0.25)
            ax.legend(fontsize="small", loc="upper right")
            ax.set_title(
                f"{panel_title}\n{record['date_label']} | peak={peak_flux:.3g} | median S/N={record['median_snr']:.2f}",
                fontsize=11,
            )

        ylim = recon_ylim(*line_flux_ranges)
        if ylim is not None:
            for ax in axes:
                if ax.axison:
                    ax.set_ylim(*ylim)

        fig.suptitle(
            f"TARGETID {target_id} | {redshift_label} | {line_name} lowest vs highest peak flux",
            fontsize=14,
        )
        plt.tight_layout(rect=(0, 0, 1, 0.95))

        line_slug = make_line_slug(line_name)
        out_path = os.path.join(out_dir, f"{target_id}_KEPT_{line_slug}_lowest_highest_peak_flux.png")
        plt.savefig(out_path, dpi=220)
        plt.close(fig)


def plot_target_from_kept_files(target_id, kept_files):
    if len(kept_files) < min_kept_coadds_per_target:
        return {
            "target_id": str(target_id),
            "passed_flux_ratio": False,
            "peak_summary_df": pd.DataFrame(),
        }

    out_dir = os.path.join(flux_ratio_pass_root, str(target_id))

    cmap = plt.get_cmap("tab20")
    colors = [cmap(index % 20) for index in range(len(kept_files))]
    redshift = load_catalog_redshift_map().get(str(target_id))
    redshift_label = f"z={redshift:.4f}" if redshift is not None and np.isfinite(redshift) else "z=unknown"

    fig, ax = plt.subplots(figsize=(14, 6))
    peak_table_rows = []
    all_recon_flux_plotted = []
    kept_file_records = []

    for index, (path, median_snr) in enumerate(kept_files):
        plot_data = load_recon_obs_arrays(path)
        if plot_data is None:
            continue

        wave_rec = plot_data["wave_rec"]
        flux_rec = plot_data["flux_rec"]
        wave_obs = plot_data["wave_obs"]
        flux_obs = plot_data["flux_obs"]

        color = colors[index]
        date_label = get_obs_date_label(path)
        kept_file_records.append(
            {
                "path": path,
                "median_snr": median_snr,
                "date_label": date_label,
            }
        )

        if wave_obs.size > 0:
            ax.plot(wave_obs, flux_obs, color=color, lw=0.3, alpha=0.08, zorder=1)
            wave_obs_c, flux_obs_c = coarse_bin(wave_obs, flux_obs, n_points=bin_points)
            ax.plot(
                wave_obs_c,
                flux_obs_c,
                color=color,
                lw=1.8,
                alpha=0.60,
                zorder=2,
                linestyle="--",
                label=f"{date_label} (obs coarse, kept S/N={median_snr:.2f})",
            )

            obs_lohi = finite_minmax(wave_obs)
            if obs_lohi is not None:
                obs_lo, obs_hi = obs_lohi
                keep_rec = mask_to_range(wave_rec, obs_lo, obs_hi)
                wave_rec_plot, flux_rec_plot = wave_rec[keep_rec], flux_rec[keep_rec]
            else:
                wave_rec_plot, flux_rec_plot = wave_rec, flux_rec
        else:
            wave_rec_plot, flux_rec_plot = wave_rec, flux_rec

        ax.plot(
            wave_rec_plot,
            flux_rec_plot,
            color=color,
            lw=2.4,
            alpha=0.50,
            zorder=3,
            label=f"{date_label} (recon, kept S/N={median_snr:.2f})",
        )

        good_recon = np.isfinite(flux_rec_plot)
        all_recon_flux_plotted.extend(flux_rec_plot[good_recon])

        flux_smoothed = smooth_flux(flux_rec_plot, width=smooth_width)
        for line_name, cfg in line_cfg.items():
            peak_w, peak_f = find_local_peak_in_window(
                wave_rec_plot,
                flux_smoothed,
                cfg["center"],
                cfg["window"],
            )
            if np.isfinite(peak_w):
                ax.scatter(
                    [peak_w],
                    [peak_f],
                    color=cfg["color"],
                    s=20,
                    zorder=10,
                    edgecolors="black",
                    linewidth=0.5,
                )
                peak_table_rows.append(
                    {
                        "TARGETID": str(target_id),
                        "OBS_DATE": date_label,
                        "MEDIAN_SNR": median_snr,
                        "LINE": line_name,
                        "PEAK_WAVE": peak_w,
                        "PEAK_FLUX": peak_f,
                    }
                )

    for _, cfg in line_cfg.items():
        ax.axvline(cfg["center"], color=cfg["color"], linestyle=":", lw=1.0, alpha=0.5)

    ax.set_title(
        f"TARGETID {target_id} | {redshift_label} | KEPT ONLY (median S/N >= {snr_cut})",
        fontsize=14,
    )
    ax.set_xlabel("Rest Wavelength (A)", fontsize=13)
    ax.set_ylabel("Flux", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xlim(*fixed_xlim)

    unified_ylim = None
    if all_recon_flux_plotted:
        all_recon_flux_plotted = np.array(all_recon_flux_plotted)
        ymin = np.min(all_recon_flux_plotted)
        ymax = np.max(all_recon_flux_plotted)
        ymin_padded = ymin - abs(ymin) * 0.3
        ymax_padded = ymax + abs(ymax) * 0.3
        ax.set_ylim(ymin_padded, ymax_padded)
        unified_ylim = (ymin_padded, ymax_padded)

    peak_df = pd.DataFrame(peak_table_rows)
    peak_summary_df = summarize_peak_flux_changes(target_id, peak_df)
    passed_flux_ratio = passes_flux_ratio_cut(peak_summary_df)

    if not passed_flux_ratio:
        plt.close(fig)
        return {
            "target_id": str(target_id),
            "passed_flux_ratio": False,
            "peak_summary_df": peak_summary_df,
        }

    os.makedirs(out_dir, exist_ok=True)

    ax.grid(alpha=0.25)
    ax.legend(fontsize="small", loc="upper right", ncol=1)
    plt.tight_layout()

    plot1_path = os.path.join(out_dir, f"{target_id}_KEPT_unified_recon_obs_lines.png")
    plt.savefig(plot1_path, dpi=220)
    plt.close(fig)

    if not peak_df.empty:
        peak_csv = os.path.join(out_dir, f"{target_id}_KEPT_emission_peaks.csv")
        peak_df.to_csv(peak_csv, index=False)

    if not peak_summary_df.empty:
        peak_summary_csv = os.path.join(out_dir, f"{target_id}_KEPT_emission_peak_flux_minmax_summary.csv")
        peak_summary_df.to_csv(peak_summary_csv, index=False)

    save_low_high_line_plots(target_id, out_dir, kept_file_records, peak_summary_df, redshift_label)

    usable_files = []
    for path, median_snr in kept_files:
        plot_data = load_recon_obs_arrays(path)
        if plot_data is None:
            continue
        usable_files.append((path, median_snr))

    if usable_files:
        n_cols = 2
        n_rows = math.ceil(len(usable_files) / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), squeeze=False)

        for index, (path, median_snr) in enumerate(usable_files):
            row, column = divmod(index, n_cols)
            ax = axes[row][column]

            plot_data = load_recon_obs_arrays(path)
            if plot_data is None:
                ax.axis("off")
                continue

            wave_obs = plot_data["wave_obs"]
            flux_obs = plot_data["flux_obs"]
            wave_rec = plot_data["wave_rec"]
            flux_rec = plot_data["flux_rec"]

            if wave_obs.size > 0:
                ax.plot(wave_obs, flux_obs, color="gray", alpha=0.12, lw=0.4, label="Obs raw")
                wave_obs_c, flux_obs_c = coarse_bin(wave_obs, flux_obs, n_points=bin_points)
                ax.plot(
                    wave_obs_c,
                    flux_obs_c,
                    color="tab:blue",
                    alpha=0.70,
                    lw=2.0,
                    label="Obs coarse",
                )
            else:
                wave_rec, flux_rec = wave_rec, flux_rec

            ax.plot(
                wave_rec,
                flux_rec,
                color="tab:red",
                alpha=0.50,
                lw=2.0,
                label="Recon",
            )

            if unified_ylim is not None:
                ax.set_ylim(unified_ylim)
            else:
                diag_ylim = recon_ylim(flux_rec)
                if diag_ylim is not None:
                    ax.set_ylim(*diag_ylim)

            ax.set_xlim(*fixed_xlim)
            ax.set_xlabel("Rest Wavelength (A)", fontsize=11)
            ax.set_ylabel("Flux", fontsize=11)
            ax.tick_params(axis="both", labelsize=10)
            ax.grid(alpha=0.25)
            ax.legend(fontsize="small", loc="upper right")
            ax.set_title(
                f"{get_obs_date_label(path)} | median S/N={median_snr:.2f} [KEPT]",
                fontsize=11,
            )

        for index in range(len(usable_files), n_rows * n_cols):
            row, column = divmod(index, n_cols)
            fig.delaxes(axes[row][column])

        fig.suptitle(f"TARGETID {target_id} | {redshift_label}", fontsize=14)
        plt.tight_layout(rect=(0, 0, 1, 0.97))
        diag_png = os.path.join(out_dir, f"{target_id}_KEPT_diagnostic_grid.png")
        plt.savefig(diag_png, dpi=200)
        plt.close(fig)

    return {
        "target_id": str(target_id),
        "passed_flux_ratio": True,
        "peak_summary_df": peak_summary_df,
    }


def main():
    os.makedirs(latent_out_dir, exist_ok=True)
    os.makedirs(flux_ratio_pass_root, exist_ok=True)

    print("=" * 70)
    print("Stage 0: Redshift filter first")
    print(f"Catalog: {catalog_csv}")
    print(f"Condition: Z > {z_min}")
    print("=" * 70)

    redshift_eligible_ids, catalog_rows = load_redshift_eligible_target_ids()
    print(f"Catalog rows loaded: {catalog_rows}")
    print(f"Targets passing redshift filter: {len(redshift_eligible_ids)}")

    print("\n" + "=" * 70)
    print("Stage 1: S/N screening")
    print(f"S/N cut: median S/N >= {snr_cut}")
    print(f"Min kept coadds per target: {min_kept_coadds_per_target}")
    print("=" * 70)

    snr_ok_map, snr_stats = screen_targets_by_snr(
        allowed_target_ids=redshift_eligible_ids,
        min_kept=min_kept_coadds_per_target,
    )
    snr_ok_targets = set(snr_ok_map.keys())

    print(f"Targets scanned in S/N stage: {snr_stats['targets_scanned']}")
    print(f"Targets passing S/N stage: {snr_stats['targets_with_min_kept']}")
    print(f"Targets removed by S/N min-coadd rule: {snr_stats['targets_removed_lt_min_kept']}")

    print("\n" + "=" * 70)
    print("Stage 2: Latent filter")
    print("Condition: n_latents_exceed_p95 > 0")
    print("=" * 70)

    _, selected_targets_df = compute_target_latent_counts(allowed_target_ids=redshift_eligible_ids)
    latent_ok_targets = set(selected_targets_df["TARGETID"].tolist())
    print(f"Targets passing latent filter: {len(latent_ok_targets)}")

    final_targets = sorted(snr_ok_targets.intersection(latent_ok_targets))

    print("\n" + "=" * 70)
    print("Stage 3: Flux-ratio cut and plotting")
    print(f"Targets after redshift + S/N + latent: {len(final_targets)}")
    print(
        "Condition: any of "
        f"{', '.join(flux_ratio_lines)} has FLUX_DIFF_OVER_LOWEST > {flux_ratio_cut}"
    )
    print("=" * 70)

    total_plotted_targets = 0
    total_plotted_kept_coadds = 0
    total_plotted_rejected_low = 0
    total_plotted_invalid = 0
    total_flux_ratio_kept = 0
    total_flux_ratio_discarded = 0
    peak_summary_frames = []
    flux_ratio_inventory_rows = []

    for target_id in final_targets:
        summary = snr_ok_map[target_id]
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
                "INVALID_SNR": summary["invalid"],
                "TOTAL_RECON": summary["total_recon"],
                "PASSED_FLUX_RATIO_CUT": bool(passed_flux_ratio),
            }
        )

        if not passed_flux_ratio:
            total_flux_ratio_discarded += 1
            print(
                f"Discarding {target_id}: kept={len(kept_files)}, "
                f"low={summary['rejected_low']}, invalid={summary['invalid']}, total={summary['total_recon']}"
            )
            continue

        total_flux_ratio_kept += 1
        total_plotted_targets += 1
        total_plotted_kept_coadds += len(kept_files)
        total_plotted_rejected_low += summary["rejected_low"]
        total_plotted_invalid += summary["invalid"]

        print(
            f"Plotting {target_id}: kept={len(kept_files)}, "
            f"low={summary['rejected_low']}, invalid={summary['invalid']}, total={summary['total_recon']}"
        )

    if peak_summary_frames:
        combined_peak_summary_df = pd.concat(peak_summary_frames, ignore_index=True)
        combined_peak_summary_df = combined_peak_summary_df.sort_values(["TARGETID", "LINE"]).reset_index(drop=True)
        combined_peak_summary_csv = os.path.join(
            flux_ratio_pass_root,
            "recon_emission_peak_flux_minmax_summary.csv",
        )
        combined_peak_summary_df.to_csv(combined_peak_summary_csv, index=False)
        print(f"Saved: {combined_peak_summary_csv}")

    if flux_ratio_inventory_rows:
        flux_ratio_inventory_df = pd.DataFrame(flux_ratio_inventory_rows)
        flux_ratio_inventory_csv = os.path.join(
            flux_ratio_pass_root,
            "flux_ratio_cut_inventory.csv",
        )
        flux_ratio_inventory_df.to_csv(flux_ratio_inventory_csv, index=False)
        print(f"Saved: {flux_ratio_inventory_csv}")

    print("\n" + "=" * 70)
    print("Pipeline complete")
    print(f"Targets kept by flux-ratio cut: {total_flux_ratio_kept}")
    print(f"Targets discarded by flux-ratio cut: {total_flux_ratio_discarded}")
    print(f"Final plotted targets: {total_plotted_targets}")
    print(f"Final plotted kept coadds: {total_plotted_kept_coadds}")
    print(f"Final plotted rejected low-S/N coadds: {total_plotted_rejected_low}")
    print(f"Final plotted invalid-S/N coadds: {total_plotted_invalid}")
    print(f"Output plot root: {flux_ratio_pass_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
