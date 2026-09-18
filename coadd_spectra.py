#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
import numpy as np
import pandas as pd
import warnings
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# DESI imports
from desispec.io import read_frame, write_spectra
from desispec.coaddition import coadd_cameras, coadd
from desispec.spectra import Spectra, stack
from desiutil.log import get_logger
from astropy.utils.metadata import MergeConflictWarning

# Silence warnings
warnings.filterwarnings('ignore', category=MergeConflictWarning, append=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Coadd spectra")
    parser.add_argument("--clq-catalog", type=str, required=True, help="Path to CLQ_candidates.csv")
    parser.add_argument("--exposures-catalog", type=str, required=True, help="Path to exposures-iron.csv")
    parser.add_argument("--target-id", type=int, required=False, help="Optional: Run only this TARGETID.")
    parser.add_argument("--coadd-cameras", action="store_true", help="Coadd cameras (b,r,z)")
    parser.add_argument("--plot", action="store_true", help="Generate a PNG plot")
    parser.add_argument("--temp-dir", type=str, default="./temp_frames", help="Directory for frame files")
    parser.add_argument("--output-base", type=str, default=".", help="Base directory for output")
    return parser.parse_args()

def plot_spectrum(spectra, output_filename, target_id, tile_id):
    """Generates a PNG plot."""
    try:
        plt.figure(figsize=(10, 6))
        plt.title(f"Target: {target_id} | Tile: {tile_id}")
        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Flux")
        for band in spectra.bands:
            wave = spectra.wave[band]
            flux = spectra.flux[band]
            if flux.ndim > 1: flux = flux[0]
            plt.plot(wave, flux, label=f"Band {band}", alpha=0.8, linewidth=0.8)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error plotting target {target_id}: {e}")

def download_file(url, local_path):
    """Downloads a file using wget if it doesn't exist."""
    if os.path.exists(local_path):
        return True

    # Ensure directory exists
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        cmd = ["wget", "-q", "-O", local_path, url]
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError:
        # Silently fail if file not found on server
        if os.path.exists(local_path):
            os.remove(local_path)
        return False

def process_target(row, df_exp, args, log):
    target_id = int(row['TARGETID'])
    tile_id = int(row['TILEID'])
    petal_loc = int(row['PETAL_LOC'])

    # Extract Night
    target_night = int(str(row['LASTNIGHT']).replace('-', ''))

    relevant_exposures = df_exp[
        (df_exp['TILEID'] == tile_id) &
        (df_exp['NIGHT'] == target_night)
    ]

    if relevant_exposures.empty:
        return False

    print(f"\n Checking Target {target_id}")
    print(f"Looking for Night: {target_night} | Tile: {tile_id} | Petal: {petal_loc}")
    print(f"Found {len(relevant_exposures)} matching exposures in Iron CSV.")

    target_dir = os.path.join(args.output_base, str(target_id))
    output_filename = f"coadd-{petal_loc}-{tile_id}-{target_night}-{target_id}.fits"
    output_path = os.path.join(target_dir, output_filename)

    if os.path.exists(output_path):
        log.info(f"Skipping {output_filename}: File exists.")
        return True

    log.info(f"Processing Target {target_id} for Night {target_night}...")

    downloaded_files = []
    bands = ['b', 'r', 'z']
    base_url = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/exposures"

    for idx, exp_row in relevant_exposures.iterrows():
        expid = int(exp_row['EXPID'])
        expid_str = f"{expid:08d}"

        for band in bands:
            fname = f"cframe-{band}{petal_loc}-{expid_str}.fits.gz"
            local_path = os.path.join(args.temp_dir, fname)

            # Construct URL: .../iron/exposures/{night}/{expid}/filename
            url = f"{base_url}/{target_night}/{expid_str}/{fname}"

            if download_file(url, local_path):
                if local_path not in downloaded_files:
                    downloaded_files.append(local_path)

    if not downloaded_files:
        log.warning(f"No files could be downloaded for Target {target_id}")
        return False

    spectra_list = []
    for filename in downloaded_files:
        try:
            fr = read_frame(filename)
            sp = Spectra(bands=[fr.meta['camera'][0]],
                         wave={fr.meta['camera'][0]: fr.wave},
                         flux={fr.meta['camera'][0]: fr.flux},
                         ivar={fr.meta['camera'][0]: fr.ivar},
                         mask={fr.meta['camera'][0]: fr.mask},
                         resolution_data={fr.meta['camera'][0]: fr.resolution_data},
                         fibermap=fr.fibermap)

            sp_selected = sp.select(targets=[target_id])
            if sp_selected.num_spectra() > 0:
                spectra_list.append(sp_selected)
        except Exception:
            pass

    if not spectra_list:
        return False

    # Stack & Coadd
    spectra_b = [s for s in spectra_list if 'b' in s.bands[0]]
    spectra_r = [s for s in spectra_list if 'r' in s.bands[0]]
    spectra_z = [s for s in spectra_list if 'z' in s.bands[0]]

    stacks = []
    if spectra_b: stacks.append(stack(spectra_b))
    if spectra_r: stacks.append(stack(spectra_r))
    if spectra_z: stacks.append(stack(spectra_z))

    if not stacks: return False

    combined_spectra = stacks[0]
    for s in stacks[1:]: combined_spectra.update(s)

    coadd(combined_spectra)
    if args.coadd_cameras:
        combined_spectra = coadd_cameras(combined_spectra)

    # Save Output
    os.makedirs(target_dir, exist_ok=True)
    write_spectra(output_path, combined_spectra)
    log.info(f"Saved: {output_filename}")

    if args.plot:
        png_filename = f"coadd-{petal_loc}-{tile_id}-{target_night}-{target_id}.png"
        png_path = os.path.join(target_dir, png_filename)
        plot_spectrum(combined_spectra, png_path, target_id, tile_id)

    # Cleanup
    for temp_file in downloaded_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            log.warning(f"Failed to delete temp file {temp_file}: {e}")

    return True

def main():
    args = parse_args()
    log = get_logger()

    os.makedirs(args.output_base, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True) # Create temp folder if missing

    try:
        df_clq = pd.read_csv(args.clq_catalog)
        df_exp = pd.read_csv(args.exposures_catalog)
        if df_exp['TILEID'].dtype != df_clq['TILEID'].dtype:
            df_exp['TILEID'] = df_exp['TILEID'].astype(int)
    except Exception as e:
        log.critical(f"Catalog Error: {e}")
        sys.exit(1)

    targets_to_process = df_clq
    if args.target_id:
        targets_to_process = df_clq[df_clq['TARGETID'] == args.target_id]

    success_count = 0
    for i, (index, row) in enumerate(targets_to_process.iterrows()):
        try:
            if process_target(row, df_exp, args, log):
                success_count += 1
        except Exception as e:
            log.error(f"Error on row {i}: {e}")

    log.info(f"Done! Checked {len(targets_to_process)} rows. Successfully saved {success_count}.")

if __name__ == '__main__':
    main()