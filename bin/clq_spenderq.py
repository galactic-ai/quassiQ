import sys
sys.path.append('/Users/iemotoyuni/Desktop/SpenderQ/SpenderQ_Clone/SpenderQ/src')
from spenderq.spenderq import SpenderQ
spender = SpenderQ("qso.dr1.hiz")

import numpy as np
from spenderq import desi_qso
import torch
import pandas as pd
from astropy.io import fits
# from desispec.io import read_spectra
# from desispec.io.fibermap import read_fibermap
from astropy.table import Table, vstack
import matplotlib.pyplot as plt

import glob
import os

desiQSO = desi_qso.DESI()
coadd_dir = "/Users/iemotoyuni/Desktop/SpenderQ/quassiQ/spender_qso/coadd"
coadd_files = sorted(glob.glob(os.path.join(coadd_dir, "*", "coadd-*.fits")))
len(coadd_files)
qsos = Table.read('/Users/iemotoyuni/Desktop/SpenderQ/catalog/CLQ_candidates.csv')

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
    # Process the already-prepared spectra for the current coadd_path

for coadd_path in coadd_files:
    print(f"\nProcessing: {coadd_path}")
    spec, w, z, target_id_t, norm, zerr = prepare_spectra(coadd_path, qsos)
    if len(spec) == 0:
        print("No valid spectra after filtering. Skipping.")
        continue

    coadd_tag = os.path.splitext(os.path.basename(coadd_path))[0]

    for i in range(len(target_id_t)):
        spec_i = spec[i:i+1]
        w_i = w[i:i+1]
        z_i = z[i:i+1]

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
            # skip spectra that cause ValueError/NaN issues (e.g. in LyA.identify_absorp)
            print(f"Skipping spectrum {i} due to error: {e}")
            continue

        print(f"[{i}] Latent shape: {s.shape}, recon shape: {recon.shape}")

        # Plot normalized vs reconstruction (rest-frame) and store for combined plotting
        wave_recon = spender.wave_recon()
        z0 = float(z_i[0].cpu().numpy())
        wave_rest = (desiQSO._wave_obs / (1.0 + z0)).cpu().numpy()

        # save arrays into recon_store per TARGETID, include redshift and observation date
        tid = int(target_id_t[i])
        # compute observation date (lastnight) from coadd filename
        tileid, lastnight = parse_tileid_lastnight(coadd_path)
        recon_store.setdefault(tid, {"specs": [], "recons": []})
        recon_store[tid]["specs"].append((wave_rest, spec_i[0].cpu().numpy(), z0, lastnight))
        recon_store[tid]["recons"].append((wave_recon, recon[0], z0, lastnight))

        # store only; individual saving will be done after collecting all spectra
        # (this allows consistent y-limits across all plots for the same TARGETID)

#====read latent space====
latent_list = []
target_id_list = []

for coadd_path in coadd_files:
    spec, w, z, target_id_t, norm, zerr = prepare_spectra(coadd_path, qsos)
    if len(spec) == 0:
        continue
    _, obs_night = parse_tileid_lastnight(coadd_path)

    for i in range(len(spec)):
        spec_i = spec[i:i+1]
        w_i    = w[i:i+1]
        z_i    = z[i:i+1]
        target_id_val = int(target_id_t[i])

        # Skip NaNs early
        if np.isnan(spec_i).any() or np.isnan(w_i).any() or np.isnan(z_i).any():
            print(f"Skipping TARGETID {target_id_val} (obs {obs_night}) due to NaNs")
            continue

        # Convert to torch tensors
        spec_i_tensor = torch.tensor(spec_i, dtype=torch.float32)
        w_i_tensor    = torch.tensor(w_i, dtype=torch.float32)
        z_i_tensor    = torch.tensor(z_i, dtype=torch.float32)

        # Safely run SpenderQ
        try:
            s, recon = spender.eval(spec_i_tensor, w_i_tensor, z_i_tensor)
            print(f"Successfully ran SpenderQ on TARGETID {target_id_val} (obs {obs_night})")
            latent_list.append(s.cpu().numpy().flatten())  # save latent vector
            target_id_list.append(target_id_val)
        except Exception as e:
            print(f"I couldn't run SpenderQ on TARGETID {target_id_val} (obs {obs_night}): {e}")
            continue

# Stack all latent vectors
if latent_list:
    latent = np.array(latent_list)  # shape (n_samples, latent_dim)
    target_ids_array = np.array(target_id_list)
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


# ==== create per-target plots (per-spectrum PNGs) ====
from itertools import zip_longest

for tid, data in recon_store.items():
    specs = data.get("specs", [])
    recons = data.get("recons", [])
    if len(specs) == 0 and len(recons) == 0:
        continue

    # choose the spectrum+recon pair with the largest dynamic range
    # and use its min/max as the shared y-limits for all individual plots
    n_pairs = max(len(specs), len(recons))
    best_min = None
    best_max = None
    best_range = -1.0
    for idx in range(n_pairs):
        vals = []
        if idx < len(specs):
            _, sp, _, _ = specs[idx]
            if getattr(sp, "size", 0):
                vals.append(np.asarray(sp).ravel())
        if idx < len(recons):
            _, rc, _, _ = recons[idx]
            if getattr(rc, "size", 0):
                vals.append(np.asarray(rc).ravel())
        if len(vals) == 0:
            continue
        arr = np.concatenate([v for v in vals if v.size > 0])
        lo = float(np.nanmin(arr))
        hi = float(np.nanmax(arr))
        rng = hi - lo
        if rng > best_range:
            best_range = rng
            best_min = lo
            best_max = hi

    # fallback to overall range if nothing found
    if best_min is None:
        all_vals = []
        for wv, sp, z_s, obs in specs:
            if getattr(sp, "size", 0):
                all_vals.append(sp.ravel())
        for wv, rc, z_r, obs_r in recons:
            if getattr(rc, "size", 0):
                all_vals.append(rc.ravel())
        if len(all_vals) == 0:
            continue
        arr = np.concatenate(all_vals)
        best_min = float(np.nanmin(arr))
        best_max = float(np.nanmax(arr))

    ymin = best_min
    ymax = best_max
    if ymax <= ymin:
        ymax = ymin + 1.0

    # prepare recon directory and clear previous files
    target_dir = os.path.join(coadd_dir, str(tid))
    recon_dir = os.path.join(target_dir, "recon")
    os.makedirs(recon_dir, exist_ok=True)
    if os.path.isdir(recon_dir):
        for _f in os.listdir(recon_dir):
            _p = os.path.join(recon_dir, _f)
            try:
                if os.path.isfile(_p) or os.path.islink(_p):
                    os.remove(_p)
                elif os.path.isdir(_p):
                    shutil.rmtree(_p)
            except Exception:
                pass

    # iterate over pairs of (spec, recon) and save simple per-spectrum plots
    for idx in range(max(len(specs), len(recons))):
        spec_tuple = specs[idx] if idx < len(specs) else None
        recon_tuple = recons[idx] if idx < len(recons) else None
        if spec_tuple is None:
            continue
        wv_s, sp, z_s, obs = spec_tuple
        wv_r, rc, z_r, obs_r = recon_tuple if recon_tuple is not None else (None, None, None, None)

        plt.figure(figsize=(12, 4))
        plt.plot(wv_s, sp, color="tab:blue", lw=0.7, alpha=0.9, label="Observed")
        if rc is not None:
            plt.plot(wv_r, rc, color="tab:red", lw=0.7, alpha=0.95, label="SpenderQ recon")
        plt.title(f"SpenderQ — TARGETID {tid} (obs {obs}, z={z_s:.4f})")
        plt.xlabel("Rest Wavelength (Å)")
        plt.ylabel("Flux")
        # use the shared y-limits (from the widest pair)
        plt.ylim(ymin, ymax)
        plt.grid(alpha=0.25)
        plt.legend()
        out_fname = f"{tid}_obs{obs}_z{z_s:.4f}_idx{idx}.png"
        out_path = os.path.join(recon_dir, out_fname)
        # remove existing file if present (clear before save)
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    # save three overlays per target: recon only, observation only, and both
    if len(specs) > 0 or len(recons) > 0:
        def _compute_ylim(arrays, fallback_min, fallback_max):
            valid = [np.asarray(a).ravel() for a in arrays if getattr(a, "size", 0)]
            if len(valid) == 0:
                return fallback_min, fallback_max
            vals = np.concatenate(valid)
            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))
            if hi <= lo:
                hi = lo + 1.0
            return lo, hi

        def _overlay_colors(n):
            if n <= 0:
                return []
            priority_colors = ["indianred", "steelblue", "seagreen"]
            if n <= 3:
                return priority_colors[:n]
            cmap = plt.get_cmap("viridis") if n > 10 else plt.get_cmap("tab10")
            return [cmap(i / max(1, n - 1)) for i in range(n)]

        n_obs_colors = max(1, len(specs))
        n_rec_colors = max(1, len(recons))
        obs_colors = _overlay_colors(n_obs_colors)
        rec_colors = _overlay_colors(n_rec_colors)

        # 1) reconstruction overlay
        if len(recons) > 0:
            rec_vals = [np.asarray(rc).ravel() for (_, rc, _, _) in recons if getattr(rc, "size", 0)]
            if len(rec_vals) > 0:
                rec_arr = np.concatenate(rec_vals)
                o_min = float(np.nanmin(rec_arr))
                o_max = float(np.nanmax(rec_arr))
                if o_max <= o_min:
                    o_max = o_min + 1.0
            else:
                o_min, o_max = ymin, ymax

            plt.figure(figsize=(14, 6))
            for j, (wv_r, rc, z_r, obs_r) in enumerate(recons):
                lbl = f"obs {obs_r} (z={z_r:.4f})"
                plt.plot(wv_r, rc, color=rec_colors[j % len(rec_colors)], lw=0.7, label=lbl, alpha=0.3)

            plt.title(f"All Reconstructed Spectra — TARGETID {tid}")
            plt.xlabel("Rest Wavelength (Å)")
            plt.ylabel("Flux")
            plt.ylim(o_min, o_max)
            plt.grid(alpha=0.25)
            plt.legend(loc="best", fontsize="small")
            overlay_path = os.path.join(recon_dir, f"TARGETID_{tid}_all_recons_overlay.png")
            plt.savefig(overlay_path, dpi=150, bbox_inches="tight")
            plt.close()

        # 2) observation overlay (pre-SpenderQ normalized input spectra)
        if len(specs) > 0:
            s_min, s_max = _compute_ylim([sp for (_, sp, _, _) in specs], ymin, ymax)
            plt.figure(figsize=(14, 6))
            for j, (wv_s, sp, z_s, obs_s) in enumerate(specs):
                plt.plot(wv_s, sp, color=obs_colors[j % len(obs_colors)], lw=0.7, alpha=0.30, linestyle="-", label=f"obs {obs_s} observed")
            plt.title(f"Observed Spectra Overlay — TARGETID {tid}")
            plt.xlabel("Rest Wavelength (Å)")
            plt.ylabel("Flux")
            plt.ylim(s_min, s_max)
            plt.grid(alpha=0.25)
            plt.legend(loc="best", fontsize="small")
            obs_overlay_path = os.path.join(recon_dir, f"TARGETID_{tid}_obs_overlay.png")
            plt.savefig(obs_overlay_path, dpi=150, bbox_inches="tight")
            plt.close()


