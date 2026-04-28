import sys
import csv
import argparse
from pathlib import Path
from typing import Any, Optional, cast

import matplotlib
matplotlib.use("Agg")
sys.path.append('/work2/11161/kanyuni/ls6/quassiQ/spenderq/src')
from spenderq.spenderq import SpenderQ
spender = SpenderQ("qso.dr1.hiz")

import numpy as np
from spenderq import desi_qso
import torch
import pandas as pd
from astropy.io import fits
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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


def _collect_available_target_ids(coadd_paths: list[str]) -> set[int]:
    """Collect TARGETIDs present in the current coadd directory tree."""
    target_ids: set[int] = set()
    for coadd_path in coadd_paths:
        target_dir = os.path.basename(os.path.dirname(coadd_path))
        try:
            target_ids.add(int(target_dir))
        except ValueError:
            continue
    return target_ids


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
    parser.add_argument(
        "--force-overlay",
        action="store_true",
        help=(
            "Recompute spectra that are already reconstructed/logged so overlay plots can be "
            "(re)generated for selected TARGETIDs."
        ),
    )
    return parser.parse_args()


args = get_args()

desiQSO = desi_qso.DESI()
coadd_dir = "/work2/11161/kanyuni/ls6/quassiQ/coadds"
coadd_files = sorted(glob.glob(os.path.join(coadd_dir, "*", "coadd-*.fits")))

requested_ids_raw = list(args.positional_target_ids or [])
requested_target_ids: list[str] = []
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
qsos = Table.read('/work2/11161/kanyuni/ls6/quassiQ/catalog/CLQ_candidates.csv')

# ========= Function for renormalization =============================

import re

def parse_tileid_lastnight(path):
    name = os.path.basename(path)
    # Accept optional suffixes (e.g. _tmp98511) before .fits.
    m = re.match(r"coadd-\d+-(\d+)-(\d+)-\d+(?:_[^.]+)*\.fits$", name)
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


def _spectrum_suffix(spectrum_index: int) -> str:
    return "" if int(spectrum_index) == 0 else f"_idx{int(spectrum_index)}"


def _product_stem(coadd_tag: str, target_id: int, spectrum_index: int) -> str:
    return f"{coadd_tag}_target{int(target_id)}{_spectrum_suffix(spectrum_index)}"


def _cleanup_stale_recon_products(recon_dir: Path, target_id: int, coadd_tag: str) -> None:
    """Remove stale products for one coadd/target while preserving overlay PNGs."""
    if not recon_dir.exists():
        return

    keep_names = {
        f"TARGETID_{int(target_id)}_obs_overlay.png",
        f"TARGETID_{int(target_id)}_recon_overlay.png",
    }
    prefix = f"{coadd_tag}_target{int(target_id)}"

    for child in recon_dir.iterdir():
        if child.name in keep_names:
            continue
        if not child.is_file():
            continue
        if child.name.startswith(prefix):
            child.unlink(missing_ok=True)


def _save_target_overlays(target_id: int, store: dict[str, list[tuple[np.ndarray, np.ndarray, str]]], recon_dir: Path) -> None:
    """Save per-target overlay plots for observed, reconstructed, and combined spectra."""
    obs_entries = store.get("obs", [])
    recon_entries = store.get("recon", [])

    if len(obs_entries) == 0 and len(recon_entries) == 0:
        return

    recon_dir.mkdir(parents=True, exist_ok=True)

    def _plot_overlay(
        entries: list[tuple[np.ndarray, np.ndarray, str]],
        title: str,
        out_path: Path,
        y_label: str,
    ) -> None:
        if len(entries) == 0:
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        for wave_values, flux_values, label in entries:
            ax.plot(wave_values, flux_values, lw=0.8, alpha=0.35, label=label)

        ax.set_title(title)
        ax.set_xlabel("Rest Wavelength (A)")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize="small")
        fig.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    _plot_overlay(
        obs_entries,
        f"Observed Spectra Overlay - TARGETID {target_id}",
        recon_dir / f"TARGETID_{target_id}_obs_overlay.png",
        "Flux",
    )

    _plot_overlay(
        recon_entries,
        f"Reconstructed Spectra Overlay - TARGETID {target_id}",
        recon_dir / f"TARGETID_{target_id}_recon_overlay.png",
        "Flux",
    )


def _save_recon_spectrum_fits(
    out_path: Path,
    target_id: int,
    coadd_tag: str,
    spectrum_index: int,
    z_value: float,
    wave_obs: np.ndarray,
    wave_rest: np.ndarray,
    observed_flux: np.ndarray,
    initial_weight: np.ndarray,
    updated_weight: np.ndarray,
    wave_recon: np.ndarray,
    recon_flux: np.ndarray,
    latent_vector: np.ndarray,
) -> None:
    """Persist one reconstructed spectrum so it can be copied and plotted elsewhere."""
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["TARGETID"] = int(target_id)
    primary_hdu.header["COADDTAG"] = str(coadd_tag)
    primary_hdu.header["SPECIDX"] = int(spectrum_index)
    primary_hdu.header["Z"] = float(z_value)

    spec_cols = [
        fits.Column(name="WAVE_OBS", format="D", array=np.asarray(wave_obs, dtype=np.float64)),
        fits.Column(name="WAVE_REST", format="D", array=np.asarray(wave_rest, dtype=np.float64)),
        fits.Column(name="OBS_FLUX", format="D", array=np.asarray(observed_flux, dtype=np.float64)),
    ]
    spec_hdu = fits.BinTableHDU.from_columns(spec_cols, name="OBSERVED")

    weight_cols = [
        fits.Column(name="WAVE_OBS", format="D", array=np.asarray(wave_obs, dtype=np.float64)),
        fits.Column(name="W_INIT", format="D", array=np.asarray(initial_weight, dtype=np.float64)),
        fits.Column(name="W_UPDATED", format="D", array=np.asarray(updated_weight, dtype=np.float64)),
    ]
    weight_hdu = fits.BinTableHDU.from_columns(weight_cols, name="WEIGHT")

    recon_cols = [
        fits.Column(name="WAVE_RECON", format="D", array=np.asarray(wave_recon, dtype=np.float64)),
        fits.Column(name="RECON_FLUX", format="D", array=np.asarray(recon_flux, dtype=np.float64)),
    ]
    recon_hdu = fits.BinTableHDU.from_columns(recon_cols, name="RECON")

    latent_arr = np.asarray(latent_vector, dtype=np.float64).reshape(1, -1)
    latent_hdu = fits.ImageHDU(data=latent_arr, name="LATENT")

    hdul = fits.HDUList([primary_hdu, spec_hdu, weight_hdu, recon_hdu, latent_hdu])
    hdul.writeto(out_path, overwrite=True)

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
LATENT_OUT_DIR = Path("work2/11161/kanyuni/ls6/quassiQ/latent")
LATENT_OUT_DIR.mkdir(parents=True, exist_ok=True)

target_count = len(_collect_available_target_ids(coadd_files))
LATENT_ALL_CSV = LATENT_OUT_DIR / f"latent_all_targets_{target_count}.csv"
# Enable restart done-markers by default for resumable long runs.
ENABLE_DONE_MARKERS = True
LATENT_ALL_COLUMNS = [
    "TARGETID",
    "COADD_TAG",
    "OBS_NIGHT",
    "SPECTRUM_INDEX",
    "REDSHIFT",
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



existing_latent_keys = _load_existing_latent_keys(LATENT_ALL_CSV)
print(f"Loaded {len(existing_latent_keys)} existing latent keys from {LATENT_ALL_CSV}")
available_target_ids = _collect_available_target_ids(coadd_files)
print(f"Detected {len(available_target_ids)} TARGETIDs in active coadd_dir")
coadd_keys_in_selection = {
    (int(os.path.basename(os.path.dirname(path))), os.path.splitext(os.path.basename(path))[0])
    for path in coadd_files
}
processed_target_ids_total = {target_id for target_id, _, _ in existing_latent_keys}
processed_coadds_total = {(target_id, coadd_tag) for target_id, coadd_tag, _ in existing_latent_keys}
completed_in_selection_before = len(processed_coadds_total & coadd_keys_in_selection)
print(
    f"Resume status before run: completed_coadds={completed_in_selection_before}/{len(coadd_keys_in_selection)}, "
    f"completed_target_ids={len(processed_target_ids_total)}"
)

n_done_markers = 0
n_appended_rows = 0
n_reconstructed = 0
n_valid_coadds = 0
n_completed_coadds = 0
reconstructed_target_ids: set[int] = set()
cleaned_recon_keys: set[tuple[int, str]] = set()

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
    if ENABLE_DONE_MARKERS and os.path.exists(done_marker) and not args.force_overlay:
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

    n_valid_coadds += 1

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
    updated_target_ids_in_coadd: set[int] = set()
    for i in range(len(target_id_t)):
        spec_i = spec[i:i + 1]
        w_i = w[i:i + 1]
        z_i = z[i:i + 1]
        tid = int(target_id_t[i])
        latent_key = (tid, coadd_tag, i)
        product_stem = _product_stem(coadd_tag, tid, i)
        recon_path = os.path.join(recon_dir, f"{product_stem}.png")
        recon_fits_path = os.path.join(recon_dir, f"{product_stem}_recon.fits")
        already_processed = (
            latent_key in existing_latent_keys
            and os.path.exists(recon_path)
            and os.path.exists(recon_fits_path)
        )

        if already_processed and not args.force_overlay:
            print(f"Skip spectrum (already logged): TARGETID={tid}, coadd={coadd_tag}, idx={i}")
            continue
        if already_processed and args.force_overlay:
            print(
                "Force mode: recomputing existing spectrum "
                f"TARGETID={tid}, coadd={coadd_tag}, idx={i}"
            )

        if torch.isnan(spec_i).any() or torch.isnan(w_i).any() or torch.isnan(z_i).any():
            print(f"Skipping TARGETID {tid} ({coadd_tag}) due to NaNs")
            continue

        cleanup_key = (tid, coadd_tag)
        if cleanup_key not in cleaned_recon_keys:
            _cleanup_stale_recon_products(Path(recon_dir), tid, coadd_tag)
            cleaned_recon_keys.add(cleanup_key)
        w_initial_i = w_i.detach().clone()

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
        latent_vector = np.asarray(s.detach().cpu().numpy()).reshape(-1)
        updated_weight = w_i[0].detach().cpu().numpy()
        initial_weight = w_initial_i[0].detach().cpu().numpy()

        _save_recon_spectrum_fits(
            out_path=Path(recon_fits_path),
            target_id=tid,
            coadd_tag=coadd_tag,
            spectrum_index=i,
            z_value=z0,
            wave_obs=np.asarray(desiQSO._wave_obs, dtype=np.float64),
            wave_rest=wave_rest,
            observed_flux=obs_spec,
            initial_weight=initial_weight,
            updated_weight=updated_weight,
            wave_recon=np.asarray(wave_recon, dtype=np.float64),
            recon_flux=np.asarray(recon[0], dtype=np.float64),
            latent_vector=latent_vector,
        )

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

        obs_label = f"{coadd_tag} | obs={obs_night if obs_night is not None else '?'} | z={z0:.4f}"
        target_store = recon_store.setdefault(tid, {"obs": [], "recon": []})
        target_store["obs"].append((wave_rest, obs_spec, obs_label))
        target_store["recon"].append((wave_recon, recon[0], obs_label))
        updated_target_ids_in_coadd.add(tid)

        latent_10 = np.full(10, np.nan, dtype=np.float64)
        n_copy = min(10, latent_vector.size)
        latent_10[:n_copy] = latent_vector[:n_copy]
        latent_variance = np.nanvar(latent_10[:9]) if latent_10.size >= 9 else np.nan

        if latent_key not in existing_latent_keys:
            latent_row: dict[str, object] = {
                "TARGETID": tid,
                "COADD_TAG": coadd_tag,
                "OBS_NIGHT": obs_night if obs_night is not None else "",
                "SPECTRUM_INDEX": i,
                "REDSHIFT": z0,
                "Variance": float(latent_variance) if np.isfinite(latent_variance) else np.nan,
            }
            for j in range(10):
                value = latent_10[j]
                latent_row[f"Latent{j + 1}"] = float(value) if np.isfinite(value) else np.nan

            _append_latent_row(LATENT_ALL_CSV, latent_row)
            existing_latent_keys.add(latent_key)
            n_appended_rows += 1

        n_reconstructed += 1
        reconstructed_target_ids.add(tid)
        processed_target_ids_total.add(tid)
        processed_coadds_total.add((tid, coadd_tag))
        coadd_success = True
        print(f"Saved recon and latent row: TARGETID={tid}, coadd={coadd_tag}, idx={i}")

    for target_id in sorted(updated_target_ids_in_coadd):
        overlay_recon_dir = Path(coadd_dir) / str(target_id) / "recon"
        _save_target_overlays(target_id, recon_store[target_id], overlay_recon_dir)
        print(f"Saved incremental overlay plots for TARGETID={target_id} in {overlay_recon_dir}")

    if coadd_success and ENABLE_DONE_MARKERS:
        Path(done_marker).touch()
        n_completed_coadds += 1
        print(f"Wrote done marker: {done_marker}")
    elif coadd_success:
        n_completed_coadds += 1
        print("Done markers disabled; not writing .done file.")
    else:
        print("No successful spectra for this coadd; done marker not written.")

print("\n==== FINAL SUMMARY ====")
print(
    f"\nRun summary: reconstructed={n_reconstructed}, "
    f"unique_target_ids={len(reconstructed_target_ids)}, "
    f"valid_coadds={n_valid_coadds}, "
    f"completed_coadds_this_run={n_completed_coadds}, "
    f"appended_rows={n_appended_rows}, skipped_done_markers={n_done_markers}"
)
print(
    f"Resume status after run: completed_coadds={len(processed_coadds_total & coadd_keys_in_selection)}/{len(coadd_keys_in_selection)}, "
    f"completed_target_ids={len(processed_target_ids_total)}"
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

if LATENT_ALL_CSV.exists():
    print(f"Latent table updated: {LATENT_ALL_CSV}")
else:
    print(f"No latent table found at {LATENT_ALL_CSV}.")


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


