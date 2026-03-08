import sys
import csv
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
sys.path.append('/work/11161/kanyuni/ls6/quassiQ/spenderq/src')
from spenderq.spenderq import SpenderQ
spender = SpenderQ("qso.dr1.hiz")

import numpy as np
from spenderq import desi_qso
import torch
import pandas as pd
from astropy.io import fits
from astropy.table import Table, vstack
import matplotlib.pyplot as plt

import glob
import os


def _parse_target_ids(values):
    """Parse TARGETIDs from CLI values supporting comma/space separated inputs."""
    parsed = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                parsed.append(token)
    return sorted(set(parsed))


def get_args():
    parser = argparse.ArgumentParser(
        description="Run SpenderQ reconstruction with optional TARGETID filtering."
    )
    parser.add_argument(
        "positional_target_ids",
        nargs="*",
        help="Optional positional TARGETIDs to process.",
    )
    parser.add_argument(
        "--target-ids",
        dest="flag_target_ids",
        nargs="+",
        default=None,
        help="Optional TARGETIDs to process (space or comma separated).",
    )
    return parser.parse_args()


args = get_args()

desiQSO = desi_qso.DESI()
coadd_dir = "/work/11161/kanyuni/ls6/quassiQ/coadds"
coadd_files = sorted(glob.glob(os.path.join(coadd_dir, "*", "coadd-*.fits")))
#coadd_files = sorted(glob.glob(os.path.join(coadd_dir, "39627853670650224", "coadd-*.fits")))

requested_ids_raw = list(args.positional_target_ids or [])
if args.flag_target_ids:
    requested_ids_raw.extend(args.flag_target_ids)

if requested_ids_raw:
    requested_target_ids = _parse_target_ids(requested_ids_raw)
    requested_target_id_set = set(requested_target_ids)
    coadd_files = [
        coadd_path
        for coadd_path in coadd_files
        if os.path.basename(os.path.dirname(coadd_path)) in requested_target_id_set
    ]
    print(
        f"Filtering to {len(requested_target_ids)} TARGETIDs: {', '.join(requested_target_ids)}"
    )

if not coadd_files:
    print("No matching coadd files found for the requested TARGETIDs.")
    sys.exit(1)

if requested_ids_raw:
    print(f"Found {len(coadd_files)} coadd files for requested TARGETIDs")
else:
    print(f"Found {len(coadd_files)} coadd files across all TARGETIDs")

len(coadd_files)
qsos = Table.read('/work/11161/kanyuni/ls6/quassiQ/catalog/CLQ_candidates.csv')

# ========= Function for renormalization =============================

import re

def parse_tileid_lastnight(path):
    name = os.path.basename(path)
    m = re.match(r"coadd-\d+-(\d+)-(\d+)-\d+\.fits$", name)
    if not m:
        return None, None
    tileid = int(m.group(1))
    lastnight = int(m.group(2))
    return tileid, lastnight

rows = []
for p in coadd_files:
    tileid, lastnight = parse_tileid_lastnight(p)
    rows.append((p, tileid, lastnight))

tile_table = Table(rows=rows, names=("path", "tileid", "lastnight"))
tile_table.pprint_all()

# store per-target spectra and reconstructions
recon_store = {}
skipped_files = []

def prepare_spectra(coadd_path, qsos):
    """Prepare a single-coadd QSO spectrum file (B/R/Z) into SpenderQ inputs."""
    with fits.open(coadd_path) as hdul:
        target_id = None
        if "FIBERMAP" in hdul:
            target_id = hdul["FIBERMAP"].data["TARGETID"][0]

        # BRZ combined case
        if "BRZ_WAVELENGTH" in hdul and "BRZ_FLUX" in hdul and "BRZ_IVAR" in hdul and "BRZ_MASK" in hdul:
            wave = hdul["BRZ_WAVELENGTH"].data.astype(np.float32)
            flux = hdul["BRZ_FLUX"].data.astype(np.float32)
            ivar = hdul["BRZ_IVAR"].data.astype(np.float32)
            mask = hdul["BRZ_MASK"].data.astype(np.int16)

            if flux.ndim == 2:
                flux = flux[0]
                ivar = ivar[0]
                mask = mask[0]

            good = mask == 0
            ivar = ivar * good

            # tensors
            spec = torch.from_numpy(flux.astype(np.float32))[None, :]
            w = torch.from_numpy(ivar.astype(np.float32))[None, :]
            if target_id is None:
                raise ValueError(f"Missing TARGETID in FIBERMAP for {os.path.basename(coadd_path)}")
            target_id = torch.tensor([int(target_id)], dtype=torch.int64)

            w[:, desiQSO._skyline_mask] = 0

            # target redshift
            z = np.zeros(1, dtype=np.float32)
            zerr = np.zeros(1, dtype=np.float32)
            match = qsos[qsos["TARGETID"] == target_id.item()]
            if len(match) > 0:
                z[:] = float(match["Z"][0])
                if "ZERR" in match.colnames:
                    zerr[:] = float(match["ZERR"][0])

            z = torch.tensor(z.astype(np.float32))
            zerr = torch.tensor(zerr.astype(np.float32))

            # normalize spectra (same window)
            norm = torch.zeros(1)
            wave_rest = desiQSO._wave_obs / (1 + z[0])
            sel = (w[0] > 0) & (wave_rest > 1600) & (wave_rest < 1800)
            if sel.count_nonzero() > 0:
                norm[0] = torch.median(spec[0][sel])
            if not torch.isfinite(norm[0]):
                norm[0] = 0
            else:
                spec[0] /= norm[0]
            w[0] *= norm[0] ** 2

            return spec, w, z, target_id, norm, zerr

        # read in data (B/R/Z split case)
        _wave, _flux, _ivar, _mask, _res = {}, {}, {}, {}, {}
        for h in range(2, len(hdul)):
            ext = hdul[h].name
            if "WAVELENGTH" in ext:
                band = ext.split("_")[0].lower()
                _wave[band] = hdul[h].data
            if "FLUX" in ext:
                band = ext.split("_")[0].lower()
                _flux[band] = hdul[h].data
            if "IVAR" in ext:
                band = ext.split("_")[0].lower()
                _ivar[band] = hdul[h].data
            if "MASK" in ext:
                band = ext.split("_")[0].lower()
                _mask[band] = hdul[h].data
            if "RESOLUTION" in ext:
                band = ext.split("_")[0].lower()
                _res[band] = hdul[h].data

    missing = []
    for b in ["b", "r", "z"]:
        if b not in _wave:
            missing.append(f"{b}:WAVELENGTH")
        if b not in _flux:
            missing.append(f"{b}:FLUX")
        if b not in _ivar:
            missing.append(f"{b}:IVAR")
        if b not in _mask:
            missing.append(f"{b}:MASK")
        if b not in _res:
            missing.append(f"{b}:RESOLUTION")
    if missing:
        raise ValueError(f"Missing spectral components in {os.path.basename(coadd_path)}: {', '.join(missing)}")


    # coadd to common grid (same logic)
    tolerance = 1e-4
    wave = _wave["b"]
    for b in ["b", "r", "z"]:
        wave = np.append(wave, _wave[b][_wave[b] > wave[-1] + tolerance])
    nwave = wave.size

    # alignment
    windict = {}
    number_of_overlapping_cameras = np.zeros(nwave)
    for b in ["b", "r", "z"]:
        imin = np.argmin(np.abs(_wave[b][0] - wave))
        windices = np.arange(imin, imin + len(_wave[b]), dtype=int)
        dwave = _wave[b] - wave[windices]
        if np.any(np.abs(dwave) > tolerance):
            raise ValueError(f"Input wavelength grids not aligned for band '{b}'.")
        number_of_overlapping_cameras[windices] += 1
        windict[b] = windices

    ndiag = max(_res[b].shape[1] for b in ["b", "r", "z"])
    ntarget = _flux["b"].shape[0]

    flux = np.zeros((ntarget, nwave), dtype=_flux["b"].dtype)
    ivar = np.zeros((ntarget, nwave), dtype=_ivar["b"].dtype)
    ivar_unmasked = np.zeros((ntarget, nwave), dtype=_ivar["b"].dtype)
    mask = np.zeros((ntarget, nwave), dtype=_mask["b"].dtype)
    rdata = np.zeros((ntarget, ndiag, nwave), dtype=_res["b"].dtype)

    for b in ["b", "r", "z"]:
        windices = windict[b]
        band_ndiag = _res[b].shape[1]
        for i in range(ntarget):
            ivar_unmasked[i, windices] += np.sum(_ivar[b][i], axis=0)
            ivar[i, windices] += _ivar[b][i] * (_mask[b][i] == 0)
            flux[i, windices] += _ivar[b][i] * (_mask[b][i] == 0) * _flux[b][i]
            for r in range(band_ndiag):
                rdata[i, r + (ndiag - band_ndiag) // 2, windices] += (
                    _ivar[b][i] * _res[b][i, r]
                )

            jj = number_of_overlapping_cameras[windices] == 1
            mask[i, windices[jj]] = _mask[b][i][jj]
            jj = number_of_overlapping_cameras[windices] > 1
            mask[i, windices[jj]] = mask[i, windices[jj]] & _mask[b][i][jj]

    for i in range(ntarget):
        ok = ivar[i] > 0
        if np.sum(ok) > 0:
            flux[i][ok] /= ivar[i][ok]
        ok = ivar_unmasked[i] > 0
        if np.sum(ok) > 0:
            rdata[i][:, ok] /= ivar_unmasked[i][ok]

    mask = mask.astype(bool) | (ivar <= 1e-6)
    ivar[mask] = 0

    # target redshift
    z = np.zeros(ntarget, dtype=np.float32)
    zerr = np.zeros(ntarget, dtype=np.float32)
    if target_id is not None:
        match = qsos[qsos["TARGETID"] == target_id]
        if len(match) > 0:
            z[:] = float(match["Z"][0])
            if "ZERR" in match.colnames:
                zerr[:] = float(match["ZERR"][0])

    # tensors
    spec = torch.from_numpy(flux.astype(np.float32))
    w = torch.from_numpy(ivar.astype(np.float32))
    if target_id is None:
        raise ValueError(f"Missing TARGETID in FIBERMAP for {os.path.basename(coadd_path)}")
    target_id = torch.tensor([int(target_id)], dtype=torch.int64)

    w[:, desiQSO._skyline_mask] = 0
    z = torch.tensor(z.astype(np.float32))
    zerr = torch.tensor(zerr.astype(np.float32))

    # normalize spectra (same window)
    norm = torch.zeros(ntarget)
    for i in range(ntarget):
        wave_rest = desiQSO._wave_obs / (1 + z[i])
        sel = (w[i] > 0) & (wave_rest > 1600) & (wave_rest < 1800)
        if sel.count_nonzero() > 0:
            norm[i] = torch.median(spec[i][sel])
        if not torch.isfinite(norm[i]):
            norm[i] = 0
        else:
            spec[i] /= norm[i]
        w[i] *= norm[i] ** 2

    keep = (spec.isfinite().sum(axis=-1) == nwave).numpy()
    print("keep: %d / %d" % (keep.sum(), len(keep)))
    return spec[keep], w[keep], z[keep], target_id[keep], norm[keep], zerr[keep]

#====

def _load_raw_single_coadd(coadd_path):
    with fits.open(coadd_path) as hdul:
        target_id = None
        if "FIBERMAP" in hdul:
            target_id = int(hdul["FIBERMAP"].data["TARGETID"][0])

        # BRZ combined case
        if "BRZ_WAVELENGTH" in hdul and "BRZ_FLUX" in hdul and "BRZ_IVAR" in hdul and "BRZ_MASK" in hdul:
            wave = hdul["BRZ_WAVELENGTH"].data.astype(np.float32)
            flux = hdul["BRZ_FLUX"].data[0].astype(np.float32)
            ivar = hdul["BRZ_IVAR"].data[0].astype(np.float32)
            mask = hdul["BRZ_MASK"].data[0].astype(np.int16)

            good = mask == 0
            ivar = ivar * good
            return target_id, wave, flux, ivar

        _wave, _flux, _ivar, _mask = {}, {}, {}, {}

        # pass 1: discover by name
        for hdu in hdul:
            ext = (hdu.name or "").upper()
            if ext.endswith("_WAVELENGTH"):
                band = ext.split("_")[0].lower()
                _wave[band] = hdu.data.astype(np.float32)
            elif ext.endswith("_FLUX"):
                band = ext.split("_")[0].lower()
                _flux[band] = hdu.data[0].astype(np.float32)
            elif ext.endswith("_IVAR"):
                band = ext.split("_")[0].lower()
                _ivar[band] = hdu.data[0].astype(np.float32)
            elif ext.endswith("_MASK"):
                band = ext.split("_")[0].lower()
                _mask[band] = hdu.data[0].astype(np.int16)

        # pass 2: direct lookup fallback
        if len(_wave) == 0:
            for band in ["B", "R", "Z"]:
                wkey, fkey, ikey, mkey = f"{band}_WAVELENGTH", f"{band}_FLUX", f"{band}_IVAR", f"{band}_MASK"
                if wkey in hdul and fkey in hdul and ikey in hdul and mkey in hdul:
                    b = band.lower()
                    _wave[b] = hdul[wkey].data.astype(np.float32)
                    _flux[b] = hdul[fkey].data[0].astype(np.float32)
                    _ivar[b] = hdul[ikey].data[0].astype(np.float32)
                    _mask[b] = hdul[mkey].data[0].astype(np.int16)

    missing = []
    for b in ["b", "r", "z"]:
        if b not in _wave:
            missing.append(f"{b}:WAVELENGTH")
        if b not in _flux:
            missing.append(f"{b}:FLUX")
        if b not in _ivar:
            missing.append(f"{b}:IVAR")
        if b not in _mask:
            missing.append(f"{b}:MASK")
    if missing:
        reason = f"Missing spectral components in {os.path.basename(coadd_path)}: {', '.join(missing)}"
        print(f"Skipping file due to missing/invalid spectral data: {coadd_path}")
        print(f"  problem: {reason}")
        skipped_files.append((coadd_path, reason))
        return None, None, None, None

    # coadd to common grid
    tolerance = 1e-4
    wave = _wave["b"]
    for b in ["b", "r", "z"]:
        wave = np.append(wave, _wave[b][_wave[b] > wave[-1] + tolerance])

    windict = {}
    for b in ["b", "r", "z"]:
        imin = np.argmin(np.abs(_wave[b][0] - wave))
        windict[b] = np.arange(imin, imin + len(_wave[b]), dtype=int)

    nwave = wave.size
    flux = np.zeros(nwave, dtype=np.float32)
    ivar = np.zeros(nwave, dtype=np.float32)

    for b in ["b", "r", "z"]:
        windices = windict[b]
        good = (_mask[b] == 0)
        ivar[windices] += _ivar[b] * good
        flux[windices] += _ivar[b] * good * _flux[b]

    ok = ivar > 0
    flux[ok] /= ivar[ok]

    return target_id, wave, flux, ivar


#==== incrementally process each coadd for restart-safe TACC runs =====
LATENT_OUT_DIR = Path("/work/11161/kanyuni/ls6/quassiQ/latent")
LATENT_OUT_DIR.mkdir(parents=True, exist_ok=True)

LATENT_ALL_CSV = LATENT_OUT_DIR / "latent_all_targets.csv"
LATENT_VAR_CSV = LATENT_OUT_DIR / "latent_variance_by_target.csv"
LATENT_VAR_DIV_MEAN_CSV = LATENT_OUT_DIR / "latent_variance_div_mean_by_target.csv"
LATENT_HIST_DIR = LATENT_OUT_DIR / "plot"
LATENT_HIST_DIR.mkdir(parents=True, exist_ok=True)
HIST_BIN_WIDTH = 0.5
LATENT_ALL_COLUMNS = [
    "TARGETID",
    "COADD_TAG",
    "OBS_NIGHT",
    "SPECTRUM_INDEX",
    *[f"Latent{i+1}" for i in range(10)],
    "Variance",
]


def _load_existing_latent_keys(csv_path: Path) -> set[tuple[int, str, int]]:
    if not csv_path.exists():
        return set()

    keys: set[tuple[int, str, int]] = set()
    with csv_path.open("r", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            return keys

        required = {"TARGETID", "COADD_TAG", "SPECTRUM_INDEX"}
        if not required.issubset(set(reader.fieldnames)):
            print(
                "Existing latent_all_targets.csv does not have "
                "COADD_TAG/SPECTRUM_INDEX; dedup by coadd is disabled for this run."
            )
            return keys

        for row in reader:
            try:
                target_id = int(row["TARGETID"])
                coadd_tag = str(row["COADD_TAG"])
                spectrum_index = int(row["SPECTRUM_INDEX"])
            except (KeyError, TypeError, ValueError):
                continue
            keys.add((target_id, coadd_tag, spectrum_index))

    return keys


def _append_latent_row(csv_path: Path, row: dict[str, object]) -> None:
    write_header = (not csv_path.exists()) or csv_path.stat().st_size == 0
    with csv_path.open("a", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=LATENT_ALL_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _latent_to_fixed_width(latent_vec: np.ndarray, width: int = 10) -> np.ndarray:
    fixed = np.full(width, np.nan, dtype=np.float64)
    n_copy = min(width, latent_vec.size)
    fixed[:n_copy] = latent_vec[:n_copy]
    return fixed


def _update_variance_products_from_latent_table(
    latent_csv: Path,
    variance_csv: Path,
    variance_div_mean_csv: Path,
    hist_dir: Path,
) -> int:
    """Recompute variance products and refresh per-latent histograms from abs(variance/mean)."""
    if not latent_csv.exists():
        return 0

    df_all = pd.read_csv(latent_csv)
    latent_cols_present = [f"Latent{i + 1}" for i in range(10) if f"Latent{i + 1}" in df_all.columns]
    if len(df_all) == 0 or not latent_cols_present:
        return 0

    df_summary = (
        df_all.groupby("TARGETID", as_index=False)[latent_cols_present]
        .var(ddof=0)
        .rename(columns={col: f"{col}_var" for col in latent_cols_present})
    )
    df_summary.to_csv(variance_csv, index=False)

    df_means = (
        df_all.groupby("TARGETID", as_index=False)[latent_cols_present]
        .mean()
        .rename(columns={col: f"{col}_mean" for col in latent_cols_present})
    )

    df_ratio = df_summary.merge(df_means, on="TARGETID", how="inner")
    ratio_columns: list[str] = []
    for latent_col in latent_cols_present:
        var_col = f"{latent_col}_var"
        mean_col = f"{latent_col}_mean"
        ratio_col = f"{var_col}_over_mean_abs"
        ratio_columns.append(ratio_col)

        var_vals = np.abs(df_ratio[var_col].to_numpy(dtype=np.float64))
        mean_vals = np.abs(df_ratio[mean_col].to_numpy(dtype=np.float64))
        ratio_vals = np.divide(
            var_vals,
            mean_vals,
            out=np.full_like(var_vals, np.nan, dtype=np.float64),
            where=mean_vals != 0,
        )
        df_ratio[ratio_col] = ratio_vals

    ratio_out_cols = ["TARGETID", *ratio_columns]
    df_ratio[ratio_out_cols].to_csv(variance_div_mean_csv, index=False)

    created_histograms = 0
    for ratio_col in ratio_columns:
        if ratio_col not in df_ratio.columns:
            continue

        values = df_ratio[ratio_col].dropna().to_numpy(dtype=np.float64)
        if values.size == 0:
            continue

        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
            bin_edges = np.arange(vmin, vmax + HIST_BIN_WIDTH, HIST_BIN_WIDTH)
            if bin_edges.size < 2:
                bins_for_hist = "auto"
            else:
                bins_for_hist = bin_edges.tolist()
        else:
            bins_for_hist = "auto"

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(values, bins=bins_for_hist, color="tab:blue", alpha=0.85, edgecolor="black")
        ax.set_title(f"Histogram of {ratio_col} across TARGETIDs")
        ax.set_xlabel(ratio_col)
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)
        fig.tight_layout()

        hist_path = hist_dir / f"{ratio_col}_hist.png"
        fig.savefig(hist_path, dpi=150)
        plt.close(fig)
        created_histograms += 1

    return created_histograms


existing_latent_keys = _load_existing_latent_keys(LATENT_ALL_CSV)
print(f"Loaded {len(existing_latent_keys)} existing latent keys from {LATENT_ALL_CSV}")

n_done_markers = 0
n_appended_rows = 0
n_reconstructed = 0

for coadd_path in coadd_files:
    print(f"\nProcessing: {coadd_path}")
    target_dir = os.path.basename(os.path.dirname(coadd_path))
    coadd_tag = os.path.splitext(os.path.basename(coadd_path))[0]
    tileid, obs_night = parse_tileid_lastnight(coadd_path)
    tile_txt = f"TILEID={tileid}" if tileid is not None else "TILEID=?"
    night_txt = f"LASTNIGHT={obs_night}" if obs_night is not None else "LASTNIGHT=?"

    processed_dir = os.path.join(coadd_dir, target_dir, "processed_spectra")
    recon_dir = os.path.join(coadd_dir, target_dir, "recon")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)

    done_marker = os.path.join(recon_dir, f"{coadd_tag}.done")
    if os.path.exists(done_marker):
        print(f"Skip coadd (done marker exists): {done_marker}")
        n_done_markers += 1
        continue

    try:
        raw_target_id, wave, flux, ivar = _load_raw_single_coadd(coadd_path)
        if wave is None:
            continue
        spec, w, z, target_id_t, norm, zerr = prepare_spectra(coadd_path, qsos)
    except Exception as e:
        print(f"Skipping file due to missing/invalid spectral data: {coadd_path}")
        print(f"  problem: {e}")
        skipped_files.append((coadd_path, str(e)))
        continue

    if len(spec) == 0:
        print("No valid spectra after filtering. Skipping.")
        continue

    raw_norm_path = os.path.join(processed_dir, f"{coadd_tag}_raw_norm.png")
    if not os.path.exists(raw_norm_path):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        axes[0].plot(wave, flux, color="gray", lw=0.8)
        axes[0].set_title(f"Raw spectrum (TARGETID={raw_target_id}, {tile_txt}, {night_txt})")
        axes[0].set_xlabel("Observed Wavelength (Å)")
        axes[0].set_ylabel("Flux")
        axes[0].grid(alpha=0.3)

        axes[1].plot(desiQSO._wave_obs, spec[0].numpy(), color="tab:blue", lw=0.8)
        axes[1].set_title(
            f"Normalized spectrum (TARGETID={int(target_id_t[0])}, z={float(z[0]):.4f}, {tile_txt}, {night_txt})"
        )
        axes[1].set_xlabel("Observed Wavelength (Å)")
        axes[1].set_ylabel("Normalized Flux")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(raw_norm_path, dpi=150)
        plt.close()

    coadd_success = False
    for i in range(len(target_id_t)):
        spec_i = spec[i:i + 1]
        w_i = w[i:i + 1]
        z_i = z[i:i + 1]
        tid = int(target_id_t[i])
        latent_key = (tid, coadd_tag, i)
        recon_path = os.path.join(recon_dir, f"{coadd_tag}_target{tid}_idx{i}.png")

        if latent_key in existing_latent_keys and os.path.exists(recon_path):
            print(f"Skip spectrum (already reconstructed and logged): TARGETID={tid}, coadd={coadd_tag}, idx={i}")
            continue

        if torch.isnan(spec_i).any() or torch.isnan(w_i).any() or torch.isnan(z_i).any():
            print(f"Skipping TARGETID {tid} ({coadd_tag}) due to NaNs")
            continue

        try:
            s, recon = spender.eval(spec_i, w_i, z_i)
        except IndexError as e:
            print(f"IndexError in Lyα absorption for index {i}: {e}")
            print("Falling back to encode/decode without Lyα masking...")
            with torch.no_grad():
                spender.models[0][0].eval()
                s = spender.models[0][0].encode(spec_i)
                recon = np.array(spender.models[0][0].decode(s))
        except Exception as e:
            print(f"Skipping spectrum {i} due to error: {e}")
            continue

        wave_recon = spender.wave_recon()
        z0 = float(z_i[0].cpu().numpy())
        wave_rest = (desiQSO._wave_obs / (1.0 + z0)).cpu().numpy()
        obs_spec = spec_i[0].cpu().numpy()

        # Save each reconstruction immediately so walltime interruptions keep prior outputs.
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(wave_rest, obs_spec, color="tab:blue", lw=0.7, alpha=0.9, label="Observed")
        ax.plot(wave_recon, recon[0], color="tab:red", lw=0.7, alpha=0.95, label="SpenderQ recon")
        ax.set_title(f"SpenderQ - TARGETID {tid} ({coadd_tag}, z={z0:.4f})")
        ax.set_xlabel("Rest Wavelength (Å)")
        ax.set_ylabel("Flux")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(recon_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        latent_vector = np.asarray(s.detach().cpu().numpy()).reshape(-1)
        latent_10 = _latent_to_fixed_width(latent_vector, width=10)
        latent_variance = np.nanvar(latent_10[:9]) if latent_10.shape[0] >= 9 else np.nan

        if latent_key not in existing_latent_keys:
            latent_row: dict[str, object] = {
                "TARGETID": tid,
                "COADD_TAG": coadd_tag,
                "OBS_NIGHT": obs_night if obs_night is not None else "",
                "SPECTRUM_INDEX": i,
                "Variance": float(latent_variance) if np.isfinite(latent_variance) else np.nan,
            }
            for j in range(10):
                value = latent_10[j]
                latent_row[f"Latent{j + 1}"] = float(value) if np.isfinite(value) else np.nan

            _append_latent_row(LATENT_ALL_CSV, latent_row)
            existing_latent_keys.add(latent_key)
            n_appended_rows += 1
            n_hist = _update_variance_products_from_latent_table(
                LATENT_ALL_CSV,
                LATENT_VAR_CSV,
                LATENT_VAR_DIV_MEAN_CSV,
                LATENT_HIST_DIR,
            )
            print(
                f"Updated variance tables and {n_hist} histogram(s) after append: "
                f"{LATENT_VAR_CSV}, {LATENT_VAR_DIV_MEAN_CSV}, {LATENT_HIST_DIR}"
            )

        n_reconstructed += 1
        coadd_success = True
        print(f"Saved recon and latent row: TARGETID={tid}, coadd={coadd_tag}, idx={i}")

    if coadd_success:
        Path(done_marker).touch()
        print(f"Wrote done marker: {done_marker}")
    else:
        print("No successful spectra for this coadd; done marker not written.")

print(
    f"\nRun summary: reconstructed={n_reconstructed}, "
    f"appended_rows={n_appended_rows}, skipped_done_markers={n_done_markers}"
)

if skipped_files:
    print("\n=== Skipped files summary ===")
    seen = set()
    for path, reason in skipped_files:
        key = (path, reason)
        if key in seen:
            continue
        seen.add(key)
        print(f"- {path}")
        print(f"  problem: {reason}")

# Ensure summary products are present even if this run appends no new rows.
if LATENT_ALL_CSV.exists():
    n_hist = _update_variance_products_from_latent_table(
        LATENT_ALL_CSV,
        LATENT_VAR_CSV,
        LATENT_VAR_DIV_MEAN_CSV,
        LATENT_HIST_DIR,
    )
    print(
        f"Final variance refresh complete: {LATENT_VAR_CSV}, {LATENT_VAR_DIV_MEAN_CSV}; "
        f"histograms available: {n_hist} in {LATENT_HIST_DIR}"
    )
else:
    print(f"No latent table found at {LATENT_ALL_CSV}; skipping summary products.")


def compute_target_latent_means(input_csv: Path, output_csv: Path) -> None:
    with input_csv.open("r", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("Input CSV is missing a header.")
        if "TARGETID" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a TARGETID column.")

        latent_columns = [
            column for column in reader.fieldnames if column.startswith("Latent")
        ]
        if not latent_columns:
            raise ValueError("No latent columns found. Expected columns like Latent1, Latent2, ...")

        sums_by_target: dict[str, dict[str, float]] = {}
        counts_by_target: dict[str, dict[str, int]] = {}
        obs_counts: dict[str, int] = {}

        for row in reader:
            target_id = row.get("TARGETID", "")
            if target_id == "":
                continue

            if target_id not in sums_by_target:
                sums_by_target[target_id] = {column: 0.0 for column in latent_columns}
                counts_by_target[target_id] = {column: 0 for column in latent_columns}
                obs_counts[target_id] = 0

            obs_counts[target_id] += 1

            for column in latent_columns:
                value = row.get(column, "")
                if value in (None, ""):
                    continue
                try:
                    numeric_value = float(value)
                except ValueError:
                    continue

                sums_by_target[target_id][column] += numeric_value
                counts_by_target[target_id][column] += 1

        with output_csv.open("w", newline="") as target:
            writer = csv.writer(target)
            writer.writerow(["TARGETID", "NumObservations", *latent_columns])

            for target_id in sorted(sums_by_target):
                row_values: list[str] = [target_id, str(obs_counts[target_id])]
                for column in latent_columns:
                    count = counts_by_target[target_id][column]
                    if count == 0:
                        row_values.append("")
                    else:
                        mean_value = sums_by_target[target_id][column] / count
                        row_values.append(f"{mean_value:.12f}")

                writer.writerow(row_values)


def divide_variance_by_mean(
    variance_csv: Path,
    mean_csv: Path,
    output_csv: Path,
) -> None:
    with mean_csv.open("r", newline="") as mean_source:
        mean_reader = csv.DictReader(mean_source)
        if not mean_reader.fieldnames:
            raise ValueError("Mean CSV is missing a header.")
        if "TARGETID" not in mean_reader.fieldnames:
            raise ValueError("Mean CSV must contain TARGETID.")

        mean_columns = [
            column
            for column in mean_reader.fieldnames
            if column.startswith("Latent") and column != "NumObservations"
        ]
        means_by_target: dict[str, dict[str, float]] = {}
        for row in mean_reader:
            target_id = row.get("TARGETID", "")
            if target_id == "":
                continue
            means_by_target[target_id] = {}
            for column in mean_columns:
                value = row.get(column, "")
                if value in (None, ""):
                    continue
                means_by_target[target_id][column] = float(value)

    with variance_csv.open("r", newline="") as var_source:
        var_reader = csv.DictReader(var_source)
        if not var_reader.fieldnames:
            raise ValueError("Variance CSV is missing a header.")
        if "TARGETID" not in var_reader.fieldnames:
            raise ValueError("Variance CSV must contain TARGETID.")

        variance_columns = [
            column for column in var_reader.fieldnames if column.startswith("Latent") and column.endswith("_var")
        ]
        if not variance_columns:
            raise ValueError("No variance columns found. Expected columns like Latent1_var.")

        latent_mapping = {column: column.removesuffix("_var") for column in variance_columns}
        for var_column, mean_column in latent_mapping.items():
            if mean_column not in mean_columns:
                raise ValueError(
                    f"Mismatch: variance column {var_column} has no matching mean column {mean_column}."
                )

        output_rows: list[list[str]] = []
        seen_variance_targets: set[str] = set()

        for row in var_reader:
            target_id = row.get("TARGETID", "")
            if target_id == "":
                continue
            seen_variance_targets.add(target_id)

            if target_id not in means_by_target:
                raise ValueError(f"Mismatch: TARGETID {target_id} exists in variance file but not in mean file.")

            output_row = [target_id]
            for var_column in variance_columns:
                mean_column = latent_mapping[var_column]
                var_value = row.get(var_column, "")
                mean_value = means_by_target[target_id].get(mean_column)

                if var_value in (None, "") or mean_value is None or mean_value == 0:
                    output_row.append("")
                    continue

                ratio = abs(float(var_value)) / abs(mean_value)
                output_row.append(f"{ratio:.12f}")

            output_rows.append(output_row)

    missing_in_variance = set(means_by_target.keys()) - seen_variance_targets
    if missing_in_variance:
        missing_sorted = ", ".join(sorted(missing_in_variance))
        raise ValueError(
            "Mismatch: these TARGETID values exist in mean file but not in variance file: "
            f"{missing_sorted}"
        )

    output_header = ["TARGETID", *[f"{column}_over_mean_abs" for column in variance_columns]]
    with output_csv.open("w", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(output_header)
        writer.writerows(output_rows)


def plot_latents_by_target(input_csv: Path, output_dir: Path) -> list[Path]:
    with input_csv.open("r", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError("Plot input CSV is missing a header.")
        if "TARGETID" not in reader.fieldnames:
            raise ValueError("Plot input CSV must contain TARGETID.")

        latent_columns = [
            column
            for column in reader.fieldnames
            if column.startswith("Latent")
            and not column.endswith("_var")
            and "_over_mean" not in column
        ]
        if not latent_columns:
            raise ValueError("No latent columns found to plot (expected Latent1 ... Latent10).")

        def latent_sort_key(name: str) -> tuple[int, str]:
            digits = "".join(char for char in name if char.isdigit())
            return (int(digits), name) if digits else (9999, name)

        latent_columns = sorted(latent_columns, key=latent_sort_key)

        target_ids: list[str] = []
        values_by_latent: dict[str, list[float]] = {column: [] for column in latent_columns}

        for row in reader:
            target_id = row.get("TARGETID", "")
            if target_id == "":
                continue
            target_ids.append(target_id)

            for column in latent_columns:
                value = row.get(column, "")
                if value in (None, ""):
                    values_by_latent[column].append(float("nan"))
                else:
                    values_by_latent[column].append(float(value))

    output_dir.mkdir(parents=True, exist_ok=True)
    created_files: list[Path] = []
    x_positions = list(range(len(target_ids)))
    highlight_suffixes = ("87", "25", "31", "86")
    point_colors = [
        "tab:orange" if any(target_id.endswith(suffix) for suffix in highlight_suffixes) else "tab:blue"
        for target_id in target_ids
    ]

    for column in latent_columns:
        fig, axis = plt.subplots(figsize=(12, 4))
        y_values = values_by_latent[column]
        axis.scatter(x_positions, y_values, s=24, c=point_colors)
        axis.set_title(f"{column} across TARGETIDs")
        axis.set_xlabel("TARGETID")
        axis.set_ylabel(column)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(target_ids, rotation=90, fontsize=8)
        axis.grid(axis="x", alpha=0.3)

        for x_value, y_value in zip(x_positions, y_values):
            if y_value != y_value:
                continue
            axis.annotate(
                f"{y_value:.3f}",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, 5),
                ha="center",
                fontsize=7,
            )
        fig.tight_layout()

        plot_path = output_dir / f"{column}_across_targets.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved latent plot: {plot_path}")
        plt.close(fig)
        created_files.append(plot_path)

    return created_files


# ==== create per-target latent means and latent plots ====
latent_all_csv = Path("/work/11161/kanyuni/ls6/quassiQ/latent/latent_all_targets.csv")
latent_mean_csv = Path("/work/11161/kanyuni/ls6/quassiQ/latent/latent_mean_by_target.csv")
latent_plot_dir = Path("/work/11161/kanyuni/ls6/quassiQ/latent/plot")

if latent_all_csv.exists():
    compute_target_latent_means(latent_all_csv, latent_mean_csv)
    print(f"Saved {latent_mean_csv}")
    created_files = plot_latents_by_target(latent_mean_csv, latent_plot_dir)
    print(f"Generated {len(created_files)} latent plots in {latent_plot_dir}")
else:
    print(f"Skipping latent plots: missing {latent_all_csv}")






    


