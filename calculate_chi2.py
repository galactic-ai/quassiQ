import os
import argparse
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate Chi2 for SpenderQ reconstructions")
    parser.add_argument("--input-csv", type=str, default="subset_cands.csv", help="Input filtered catalog")
    parser.add_argument("--out-csv", type=str, default="post_chi2_calc.csv", help="Output merged CSV file")
    parser.add_argument("--recon-dir", type=str, default="reconstructions", help="Base directory of SpenderQ outputs")
    parser.add_argument("--out-plot", type=str, default="chi2_hist.png", help="Output histogram image")
    parser.add_argument("--save-freq", type=int, default=500, help="How often to save progress")
    return parser.parse_args()

def calculate_chi2(fits_path):
    """Calculates the reduced chi-squared for a single SpenderQ reconstruction."""
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            
            orig_flux = data['ORIG_FLUX']
            recon_flux = data['RECON_FLUX']
            weights = data['UPDATED_WEIGHTS']
            
            # Mask where weights are 0 also removes unobserved regions
            valid = (weights > 0) & np.isfinite(orig_flux) & np.isfinite(recon_flux)
            dof = np.sum(valid)
            
            if dof == 0:
                return np.nan, 0
                
            # chi2 = sum( (obs - model)^2 * weight )
            chi2 = np.sum((orig_flux[valid] - recon_flux[valid])**2 * weights[valid])
            reduced_chi2 = chi2 / dof
            
            return reduced_chi2, dof
            
    except Exception as e:
        print(f"Error processing {fits_path}: {e}")
        return np.nan, 0

def main():
    args = parse_args()
    
    # 1. Load or Resume the Catalog
    if os.path.exists(args.out_csv):
        print(f"Resuming from existing {args.out_csv}...")
        df = pd.read_csv(args.out_csv)
    else:
        if not os.path.exists(args.input_csv):
            print(f"Error: Could not find {args.input_csv}")
            return
        print(f"Starting fresh from {args.input_csv}...")
        df = pd.read_csv(args.input_csv)
        # Add new columns
        df['REDUCED_CHI2'] = np.nan
        df['DOF'] = np.nan

    # Find rows that haven't been processed yet 
    unprocessed_mask = df['DOF'].isna()
    indices_to_process = df[unprocessed_mask].index
    
    total_rows = len(df)
    processed_count = total_rows - len(indices_to_process)
    
    print(f"Total rows: {total_rows} | Already processed: {processed_count} | Remaining: {len(indices_to_process)}\n")
    
    # 2. Iterate and process missing files
    processed_in_this_run = 0
    
    for i, idx in enumerate(indices_to_process, 1):
        row = df.loc[idx]
        tid = int(row['TARGETID'])
        
        raw_night = str(row['LASTNIGHT']).replace('-', '').split()[0]
        night = int(raw_night)
        
        fits_path = os.path.join(args.recon_dir, str(tid), str(night), f"recon_{tid}_{night}.fits")
        
        if os.path.exists(fits_path):
            rchi2, dof = calculate_chi2(fits_path)
            df.at[idx, 'REDUCED_CHI2'] = rchi2
            df.at[idx, 'DOF'] = dof
        else:
            df.at[idx, 'REDUCED_CHI2'] = np.nan
            df.at[idx, 'DOF'] = -1
            
        processed_in_this_run += 1
        
        if processed_in_this_run % args.save_freq == 0:
            print(f"Processed {processed_in_this_run}/{len(indices_to_process)} remaining rows.")
            df.to_csv(args.out_csv, index=False)
            
    # Final save
    print("Saving final completed catalog.")
    df.to_csv(args.out_csv, index=False)
    
    # 3. Generate the Histogram
    print(f"Generating histogram: {args.out_plot}")
    
    # Filter valid Chi2 values for the plot (ignore NaNs and missing files)
    valid_plot_data = df[df['REDUCED_CHI2'].notna() & (df['DOF'] > 0)]
    
    if len(valid_plot_data) > 0:
        plot_data = valid_plot_data['REDUCED_CHI2']
        
        plt.figure(figsize=(10, 6))
        plt.hist(plot_data, bins=np.linspace(0.5, 3.5, 120), color='royalblue', edgecolor='black', alpha=0.7)
        
        # Calculate percentiles for thresholding
        median_chi2 = np.median(plot_data)
        p75 = np.percentile(plot_data, 75)
        p90 = np.percentile(plot_data, 90)
        p95 = np.percentile(plot_data, 95)
        p99 = np.percentile(plot_data, 99)
        
        # Plot percentile lines
        plt.axvline(median_chi2, color='red', linestyle='solid', linewidth=2, label=f'Median: {median_chi2:.2f}')
        plt.axvline(p75, color='orange', linestyle='dashed', linewidth=1.5, label=f'75th %: {p75:.2f}')
        plt.axvline(p90, color='green', linestyle='dashed', linewidth=1.5, label=f'90th %: {p90:.2f}')
        plt.axvline(p95, color='purple', linestyle='dashed', linewidth=1.5, label=f'95th %: {p95:.2f}')
        plt.axvline(p99, color='brown', linestyle='dashed', linewidth=1.5, label=f'99th %: {p99:.2f}')
        
        plt.title('Distribution of Reduced $\\chi^2$ for SpenderQ Reconstructions')
        plt.xlabel('Reduced $\\chi^2$')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.xlim(0,3.5)
        plt.savefig(args.out_plot, dpi=150)
        plt.close()
        
    else:
        print("No valid Chi2 values found to plot.")

if __name__ == "__main__":
    main()