import os
import sys
import csv
import traceback
import torch
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from desispec.io import read_spectra
from desispec.coaddition import coadd_cameras
from desispec.interpolation import resample_flux
from scipy.interpolate import interp1d

# SpenderQ Imports
for _p in ('/work/10579/prisha/ls6/desi_project/SpenderQ_repo/src',
           '/home/jovyan/work/ls6/desi_project/SpenderQ_repo/src'):
    if os.path.isdir(_p):
        sys.path.append(_p)
        break
else:
    raise RuntimeError("SpenderQ source directory not found")
    
from spenderq.spenderq import SpenderQ
from spenderq import desi_qso


def filter_catalog():
    print("Filtering Catalog")
    df = pd.read_csv("CLQ_cands_snr.csv")
    print(f"Input: {len(df)} rows, {df['TARGETID'].nunique()} unique targets")

    # Apply SNR cut
    df = df[df['MEDIAN_SNR'] >= 2.0].copy()
    print(f"After SNR >= 2.0: {len(df)} rows, {df['TARGETID'].nunique()} unique targets")

    # Recalculate duration/counts on surviving observations
    df['LASTNIGHT'] = pd.to_datetime(df['LASTNIGHT'].astype(str), format='%Y-%m-%d')
    grp = df.groupby('TARGETID')['LASTNIGHT'].agg(['min', 'max', 'count'])
    grp['duration_days'] = (grp['max'] - grp['min']).dt.days

    # Keep only targets with >1 observation AND >= 7 days duration
    valid_tids = grp[(grp['count'] > 1) & (grp['duration_days'] >= 7)].index
    df = df[df['TARGETID'].isin(valid_tids)].copy()
    print(f"After cadence: {len(df)} rows, {df['TARGETID'].nunique()} unique targets")

    output_csv = "subset_cands.csv"
    df.to_csv(output_csv, index=False)

    print(f"Filtered down to {len(df)} total observations across {df['TARGETID'].nunique()} unique targets.")
    print(f"Saved to {output_csv}\n")
    return df
    
def run_spenderq(df, coadd_dir="output_coadds", out_dir="reconstructions"):
    print("Initializing SpenderQ")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    sq = SpenderQ("qso.dr1.hiz")
    for m in sq.models:
        m[0].to(device)
    desiQSO = desi_qso.DESI()

    wave_obs = np.asarray(desiQSO._wave_obs)
    wave_recon_rest = np.asarray(sq.wave_recon())
    n_obs = len(wave_obs)
    print(f"Observed grid: {n_obs} px | Model rest grid: {len(wave_recon_rest)} px")

    latent_csv = "latent_info.csv"
    latent_cols = ["TARGETID", "LASTNIGHT"] + [f"Latent{i+1}" for i in range(10)]

    if not os.path.exists(latent_csv):
        with open(latent_csv, 'w', newline='') as f:
            csv.writer(f).writerow(latent_cols)

    done_latents = set()
    try:
        _d = pd.read_csv(latent_csv, usecols=["TARGETID", "LASTNIGHT"])
        done_latents = set(zip(_d["TARGETID"].astype(int), _d["LASTNIGHT"].astype(int)))
    except Exception:
        pass
    print(f"Found {len(done_latents)} existing latent rows\n")

    total_rows = len(df)

    # Use enumerate to accurately track the exact row index
    for idx, (row_idx, row) in enumerate(df.iterrows(), start=1):
        tid = int(row['TARGETID'])
        night = int(row['LASTNIGHT'].strftime('%Y%m%d'))
        tile = int(row['TILEID'])
        petal = int(row['PETAL_LOC'])
        z_val = float(row['Z'])

        target_dir = os.path.join(out_dir, str(tid), str(night))
        os.makedirs(target_dir, exist_ok=True)

        fits_out = os.path.join(target_dir, f"recon_{tid}_{night}.fits")
        png_out = os.path.join(target_dir, f"recon_{tid}_{night}.png")

        # Only skip when all three outputs exist
        if os.path.exists(fits_out) and os.path.exists(png_out) and (tid, night) in done_latents:
            print(f"[{idx}/{total_rows}] Skipped {tid} on {night} (Already exists)")
            continue

        coadd_path = os.path.join(coadd_dir, str(tid), f"coadd-{petal}-{tile}-{night}-{tid}.fits")
        if not os.path.exists(coadd_path):
            print(f"[{idx}/{total_rows}] Missing coadd: {coadd_path}")
            continue

        try:
            sp = read_spectra(coadd_path)

            # If 'brz' is missing, stitch the individual cameras
            if 'brz' not in sp.bands:
                if 'b' in sp.bands and 'r' in sp.bands and 'z' in sp.bands:
                    sp = coadd_cameras(sp)
                else:
                    print(f"[{idx}/{total_rows}] Skipping {tid} on {night}: Missing 'brz' and lacking all 3 individual bands.")
                    continue

            raw_wave = sp.wave['brz']
            raw_flux = sp.flux['brz'][0]
            raw_ivar = sp.ivar['brz'][0]

            if len(raw_flux) != n_obs:
                if len(raw_wave) != len(raw_flux):
                    print(f"[{idx}/{total_rows}] Skipping {tid} on {night}: Mismatched wave/flux lengths ({len(raw_wave)} vs {len(raw_flux)})")
                    continue

                flux = interp1d(raw_wave, raw_flux, bounds_error=False, fill_value=0.0)(wave_obs)
                ivar = interp1d(raw_wave, raw_ivar, bounds_error=False, fill_value=0.0)(wave_obs)
            else:
                flux = raw_flux
                ivar = raw_ivar

            spec_t = torch.from_numpy(np.asarray(flux, dtype=np.float32))[None, :].to(device)
            w_t = torch.from_numpy(np.asarray(ivar, dtype=np.float32))[None, :].to(device)
            z_t = torch.tensor([z_val], dtype=torch.float32).to(device)

            w_t[:, desiQSO._skyline_mask] = 0
            wave_rest = wave_obs / (1 + z_val)
            sel = (w_t[0] > 0) & (torch.from_numpy(wave_rest).to(device) > 1600) & \
                  (torch.from_numpy(wave_rest).to(device) < 1800)
            norm = torch.median(spec_t[0][sel]) if sel.count_nonzero() > 0 else torch.tensor(1.0, device=device)

            if norm > 0:
                spec_t[0] /= norm
                w_t[0] *= norm ** 2

            # Count valid pixels before SpenderQ absorption masking
            pre_sq_valid_pixels = (w_t.cpu().numpy().squeeze() > 0).sum()

            with torch.no_grad():
                s, recon = sq.eval(spec_t, w_t, z_t)

                # w_t is modified in-place by SpenderQ eval()
                updated_weights = w_t.cpu().numpy().squeeze()

            # Count valid pixels after SpenderQ masks absorption
            post_sq_valid_pixels = (updated_weights > 0).sum()
            pixels_masked = pre_sq_valid_pixels - post_sq_valid_pixels

            orig_flux = spec_t.cpu().numpy().squeeze()
            latents = s.cpu().numpy().squeeze()

            recon_flux_rest = np.asarray(recon).squeeze()
            recon_flux = interp1d(
                wave_recon_rest * (1 + z_val), recon_flux_rest,
                bounds_error=False, fill_value=np.nan
            )(wave_obs)

            lengths = {len(wave_rest), len(orig_flux), len(recon_flux), len(updated_weights)}
            if len(lengths) != 1:
                print(f"[{idx}/{total_rows}] Skipping {tid} on {night}: inconsistent column lengths {lengths}")
                continue

            # Save FITS
            c1 = fits.Column(name='REST_WAVE', format='E', array=wave_rest)
            c2 = fits.Column(name='ORIG_FLUX', format='E', array=orig_flux)
            c3 = fits.Column(name='RECON_FLUX', format='E', array=recon_flux)
            c4 = fits.Column(name='UPDATED_WEIGHTS', format='E', array=updated_weights)

            hdu = fits.BinTableHDU.from_columns([c1, c2, c3, c4])
            hdu.header['TARGETID'] = tid
            hdu.header['NIGHT'] = night
            hdu.header['Z'] = z_val
            hdu.header['NORM'] = float(norm)
            hdu.writeto(fits_out, overwrite=True)

            # Save PNG
            plt.figure(figsize=(10, 4))
            plt.plot(wave_rest, orig_flux, label="Coadd", color='k', alpha=0.4, drawstyle='steps-mid')
            plt.plot(wave_rest, recon_flux, label="SpenderQ", color='r')
            plt.xlabel("Rest wavelength [A]")
            plt.ylabel("Normalized flux")
            plt.title(f"Target: {tid} | Night: {night} | z = {z_val:.3f}")
            plt.legend()
            plt.savefig(png_out, dpi=150, bbox_inches='tight')
            plt.close()

            # Append Latents 
            if (tid, night) not in done_latents:
                with open(latent_csv, 'a', newline='') as f:
                    csv.writer(f).writerow([tid, night] + list(latents))
                done_latents.add((tid, night))

            print(f"[{idx}/{total_rows}] Processed: {tid} on {night} | Masked by SpenderQ: {pixels_masked} px")

        except Exception as e:
            print(f"[{idx}/{total_rows}] Error on {tid} ({night}): {e}")
            traceback.print_exc()

if __name__ == "__main__":
    df_subset = filter_catalog()
    run_spenderq(df_subset)