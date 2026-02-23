import sys
#=================change============================
sys.path.append('/Users/iemotoyuni/Desktop/SpenderQ/SpenderQ_Clone/SpenderQ/src')

from spenderq.spenderq import SpenderQ
spender = SpenderQ("qso.dr1.hiz")
from spenderq import desi_qso
import torch
import numpy as np
import pandas as pd
from astropy.io import fits
from desispec.io import read_spectra
from desispec.io.fibermap import read_fibermap
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
import glob
import os
import re
import shutil


desiQSO = desi_qso.DESI()
#=================change============================
coadd_dir = "/Users/iemotoyuni/Desktop/SpenderQ/quassiQ/spender_qso/coadd"
qsos = Table.read('/Users/iemotoyuni/Desktop/SpenderQ/catalog/CLQ_candidates.csv')

# Collect coadd FITS from both top-level and per-target subdirectories
coadd_files = sorted(glob.glob(os.path.join(coadd_dir, "*", "coadd-*.fits")))
len(coadd_files)

# Find newly generated coadd files
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

# ========= Function for renormalization =============================

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

    required = {"b", "r", "z"}
    if not required.issubset(_wave.keys()):
        raise KeyError(f"Missing band(s): {sorted(required - set(_wave.keys()))}")


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

    required = {"b", "r", "z"}
    if not required.issubset(_wave.keys()):
        raise KeyError(f"Missing band(s): {sorted(required - set(_wave.keys()))}")

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


#====disply the normalized spectra =====
import shutil

for coadd_path in coadd_files:
    print(f"\nProcessing: {coadd_path}")

    # Raw
    target_id, wave, flux, ivar = _load_raw_single_coadd(coadd_path)

    # Normalized
    spec, w, z, target_id_t, norm, zerr = prepare_spectra(coadd_path, qsos)
    if len(spec) == 0:
        print("No valid spectra after filtering. Skipping.")
        continue

    tileid, lastnight = parse_tileid_lastnight(coadd_path)
    tile_txt = f"TILEID={tileid}" if tileid is not None else "TILEID=?"
    night_txt = f"LASTNIGHT={lastnight}" if lastnight is not None else "LASTNIGHT=?"

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    axes[0].plot(wave, flux, color="gray", lw=0.8)
    axes[0].set_title(f"Raw spectrum (TARGETID={target_id}, {tile_txt}, {night_txt})")
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

    target_dir = os.path.basename(os.path.dirname(coadd_path))
    out_dir = os.path.join(coadd_dir, target_dir, "processed_spectra")
    os.makedirs(out_dir, exist_ok=True)
    out_name = os.path.splitext(os.path.basename(coadd_path))[0]
    out_path = os.path.join(out_dir, f"{out_name}_raw_norm.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    #plt.show()

    #===RUN SpenderQ on the normalized spectrum ===
    for coadd_path in coadd_files:
    print(f"\nProcessing: {coadd_path}")
    spec, w, z, target_id_t, norm, zerr = prepare_spectra(coadd_path, qsos)
    if len(spec) == 0:
        print("No valid spectra after filtering. Skipping.")
        continue

    coadd_tag = os.path.splitext(os.path.basename(coadd_path))[0]

    for i in range(len(target_id_t)):
        spec_i = spec[i:i+1]
        z_i = z[i:i+1]

        try:
            s, recon = spender.eval(spec_i, w, z_i)
        except IndexError as e:
            print(f"IndexError in Lyα absorption for index {i}: {e}")
            print("Falling back to encode/decode without Lyα masking...")
            with torch.no_grad():
                spender.models[0][0].eval()
                s = spender.models[0][0].encode(spec_i)
                recon = np.array(spender.models[0][0].decode(s))
        except Exception as e:
            # skip spectra that cause ValueError/NaN issues (e.g. in LyA.identify_absorp)
            print(f"Skipping spectrum {i} due to error: {e}")
            continue

        print(f"[{i}] Latent shape: {s.shape}, recon shape: {recon.shape}")

        # Plot normalized vs reconstruction (rest-frame)
        wave_recon = spender.wave_recon()
        z0 = float(z_i[0].cpu().numpy())
        wave_rest = (desiQSO._wave_obs / (1.0 + z0)).cpu().numpy()

        plt.figure(figsize=(12, 4))
        plt.plot(wave_rest, spec_i[0].cpu().numpy(), color="tab:blue", lw=0.8, label="Normalized")
        plt.plot(wave_recon, recon[0], color="tab:red", lw=0.8, label="SpenderQ recon")
        plt.title(f"SpenderQ — TARGETID {int(target_id_t[i])} (z={z0:.4f})")
        plt.xlabel("Rest Wavelength (Å)")
        plt.ylabel("Flux")
        plt.grid(alpha=0.3)
        plt.legend()
        tid = int(target_id_t[i])
        target_dir = os.path.join(coadd_dir, str(tid))
        recon_dir = os.path.join(target_dir, "recon")
        os.makedirs(recon_dir, exist_ok=True)
        out_fname = f"{coadd_tag}_TARGETID_{tid}_recon.png"
        out_path = os.path.join(recon_dir, out_fname)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

#====read latent space====
latent_list = []
target_id_list = []

for coadd_path in coadd_files:
    spec, w, z, target_id_t, norm, zerr = prepare_spectra(coadd_path, qsos)
    if len(spec) == 0:
        continue

    for i in range(len(spec)):
        spec_i = spec[i:i+1]
        w_i    = w[i:i+1]
        z_i    = z[i:i+1]

        # Skip NaNs early
        if np.isnan(spec_i).any() or np.isnan(w_i).any() or np.isnan(z_i).any():
            print(f"Skipping spectrum {i} due to NaNs")
            continue

        # Convert to torch tensors
        spec_i_tensor = torch.tensor(spec_i, dtype=torch.float32)
        w_i_tensor    = torch.tensor(w_i, dtype=torch.float32)
        z_i_tensor    = torch.tensor(z_i, dtype=torch.float32)

        # Safely run SpenderQ
        try:
            s, recon = spender.eval(spec_i_tensor, w_i_tensor, z_i_tensor)
            print(f"Successfully ran SpenderQ on index {i}")
            latent_list.append(s.cpu().numpy().flatten())  # save latent vector
            target_id_list.append(int(target_id_t[i]))
        except Exception as e:
            print(f"I couldn't run SpenderQ on index {i}: {e}")
            continue

# Stack all latent vectors
if latent_list:
    latent = np.array(latent_list)  # shape (n_samples, latent_dim)
    print(f"Collected {latent.shape[0]} latent vectors of size {latent.shape[1]}")
else:
    print("No latent vectors collected — check your data!")

print(target_id_list)

# ==== download the latent space and target IDs in CSV ====
        
out_dir = "/Users/iemotoyuni/Desktop/SpenderQ/quassiQ/spender_qso/latent"
os.makedirs(out_dir, exist_ok=True)

rows = []
summary_rows = []
for targetid in np.unique(target_ids_array):
    mask = target_ids_array == targetid
    latent_for_target = latent[mask]
    if latent_for_target.size == 0:
        continue

    n_samples, latent_dim = latent_for_target.shape

    # ensure exactly 10 latent columns (truncate or pad with NaN)
    if latent_dim >= 10:
        lat10 = latent_for_target[:, :10]
    else:
        pad = np.full((n_samples, 10 - latent_dim), np.nan)
        lat10 = np.hstack([latent_for_target, pad])

    # per-row variance over first 9 latents (indices 0..8)
    var_per_row = np.full(n_samples, np.nan)
    if lat10.shape[1] >= 9:
        var_per_row = np.nanvar(lat10[:, :9], axis=1)

    # append per-spectrum rows (TARGETID, Latent1..Latent10, Variance)
    for r, v in zip(lat10, var_per_row):
        rows.append([int(targetid)] + list(r.tolist()) + [float(v) if np.isfinite(v) else np.nan])

    # per-id summary: variance across spectra for each latent dimension (use nanvar)
    var_per_dim = np.nanvar(lat10, axis=0)  # shape (10,)
    summary_rows.append([int(targetid)] + [float(x) if np.isfinite(x) else np.nan for x in var_per_dim])

# save full table (one row per spectrum)
if len(rows) > 0:
    cols = ["TARGETID"] + [f"Latent{i+1}" for i in range(10)] + ["Variance"]
    df_all = pd.DataFrame(rows, columns=cols)
    out_csv = os.path.join(out_dir, "latent_all_targets.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df_all)} rows)")
else:
    print("No latent rows to save.")

# save per-target summary (one row per TARGETID) with variances per latent dim
if len(summary_rows) > 0:
    summary_cols = ["TARGETID"] + [f"Latent{i+1}_var" for i in range(10)]
    df_summary = pd.DataFrame(summary_rows, columns=summary_cols)
    summary_csv = os.path.join(out_dir, "latent_variance_by_target.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv} ({len(df_summary)} rows)")




    

