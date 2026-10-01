import os
import argparse
import numpy as np
import pandas as pd
from desispec.io import read_spectra

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Median SNR for generated coadds")
    parser.add_argument("--clq-catalog", type=str, default="CLQ_candidates.csv", help="Path to CLQ_candidates.csv")
    parser.add_argument("--out-catalog", type=str, default="CLQ_cands_snr.csv", help="Path to save the output with SNR")
    parser.add_argument("--coadd-base-dir", type=str, default="output_coadds", help="Base directory where target ID folders are stored")
    return parser.parse_args()

def calculate_median_snr(filepath):
    """
    Reads the spectra FITS file and calculates the overall median SNR.
    SNR is calculated per pixel as: Flux * sqrt(IVAR).
    """
    try:
        sp = read_spectra(filepath)
        all_snr = []
        
        for band in sp.bands:
            flux = sp.flux[band].flatten()
            ivar = sp.ivar[band].flatten()
            
            # Mask out invalid pixels (ivar <= 0)
            valid_pixels = ivar > 0
            
            if np.any(valid_pixels):
                # Calculate SNR for valid pixels
                snr = flux[valid_pixels] * np.sqrt(ivar[valid_pixels])
                all_snr.extend(snr)
        
        if len(all_snr) > 0:
            return np.median(all_snr)
        else:
            return np.nan
            
    except Exception as e:
        print(f"Error reading/calculating SNR for {filepath}: {e}")
        return np.nan

def main():
    args = parse_args()
    
    if not os.path.exists(args.clq_catalog):
        print(f"Error: Catalog file {args.clq_catalog} not found.")
        return

    # Read the original input catalog
    df_input = pd.read_csv(args.clq_catalog)
    
    if 'MEDIAN_SNR' not in df_input.columns:
        df_input['MEDIAN_SNR'] = np.nan

    processed_count = 0
    
    # Check if the output file already exists
    if os.path.exists(args.out_catalog):
        df_existing = pd.read_csv(args.out_catalog)
        processed_count = len(df_existing)
        print(f"Found {processed_count} rows already processed in {args.out_catalog}.")
    else:
        df_input.head(0).to_csv(args.out_catalog, index=False)
        print(f"Started fresh. Created {args.out_catalog}.")

    skipped_count = processed_count
    newly_processed = 0
    missing_file_count = 0

    print(f"Total rows to process: {len(df_input) - processed_count}")

    for idx in range(processed_count, len(df_input)):
        row = df_input.iloc[idx].copy()
        
        target_id = int(row['TARGETID'])
        tile_id = int(row['TILEID'])
        petal_loc = int(row['PETAL_LOC'])
        target_night = int(str(row['LASTNIGHT']).replace('-', '').split()[0])
        
        target_dir = os.path.join(args.coadd_base_dir, str(target_id))
        filename = f"coadd-{petal_loc}-{tile_id}-{target_night}-{target_id}.fits"
        filepath = os.path.join(target_dir, filename)

        if os.path.exists(filepath):
            print(f"Calculating SNR for Target: {target_id} | Night: {target_night}.")
            
            snr_val = calculate_median_snr(filepath)
            row['MEDIAN_SNR'] = snr_val
            
            if not np.isnan(snr_val):
                newly_processed += 1
                
            print(f"Median SNR: {snr_val:.2f} | Total rows saved: {processed_count + newly_processed + missing_file_count + 1}")
        else:
            missing_file_count += 1
            print(f"Skipped: Coadd file missing for Target {target_id}. Appending as NaN.")

        pd.DataFrame([row]).to_csv(args.out_catalog, mode='a', header=False, index=False)

    print("\n Summary")
    print(f"Previously completed (skipped): {skipped_count}")
    print(f"Newly calculated & saved:       {newly_processed}")
    print(f"Skipped (coadd file missing):   {missing_file_count}")

if __name__ == "__main__":
    main()