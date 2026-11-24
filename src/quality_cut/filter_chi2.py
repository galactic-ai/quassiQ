import os
import shutil

import numpy as np
import pandas as pd


def process_catalogs_and_plots():
    input_csv = "/work/10579/prisha/ls6/desi_project/post_chi2_calc.csv"
    output_csv = "/work/11161/kanyuni/ls6/quassiQ_project/quality_cut/final_subset.csv"
    recons_dir = "/work/10579/prisha/ls6/desi_project/reconstructions"
    
    print(f"Loading {input_csv}...")
    df = pd.read_csv(input_csv, dtype={"TARGETID": "string", "LASTNIGHT": "string"})
    
    # 1. Identify rows that pass/fail the Chi-Square cut
    chi2 = pd.to_numeric(df["REDUCED_CHI2"], errors="coerce")
    valid_chi2 = np.isfinite(chi2) & chi2.ge(0)
    survived_chi2_mask = valid_chi2 & chi2.le(1.4)
    
    survived_chi2_df = df[survived_chi2_mask].copy()
    failed_chi2_df = df[~survived_chi2_mask].copy()
    
    print(f"Observations surviving chi2 <= 1.4: {len(survived_chi2_df)}")
    print(f"Observations failing chi2 cut: {len(failed_chi2_df)}")
    print(f"  Above 1.4: {(valid_chi2 & chi2.gt(1.4)).sum():,}")
    print(f"  Missing or invalid chi2: {(~valid_chi2).sum():,}")
    print(f"Targets lost at the chi2 cut: {df['TARGETID'].nunique() - survived_chi2_df['TARGETID'].nunique():,}")

    # 2. Process surviving observations: count and duration calculations
    # Convert LASTNIGHT to datetime to do date math
    nights = survived_chi2_df["LASTNIGHT"].str[:10].str.replace("-", "", regex=False)
    survived_chi2_df["LASTNIGHT_dt"] = pd.to_datetime(nights, format="%Y%m%d")
    
    # Group by target to find new min/max dates and counts
    grp = survived_chi2_df.groupby('TARGETID')['LASTNIGHT_dt'].agg(['min', 'max', 'count'])
    grp['new_duration'] = (grp['max'] - grp['min']).dt.days

    # Identify valid targets (>1 observation AND >= 7 days duration)
    valid_tids = grp[(grp['count'] > 1) & (grp['new_duration'] >= 7)].index
    
    # Filter the dataframe to only include these valid targets
    final_df = survived_chi2_df[survived_chi2_df['TARGETID'].isin(valid_tids)].copy()
    print(f"Targets lost after rechecking cadence: {len(grp) - len(valid_tids):,}")
    
    # Update the 'duration_days' column with the newly calculated durations
    duration_map = grp['new_duration'].to_dict()
    final_df['duration_days'] = final_df['TARGETID'].map(duration_map)
    
    # Drop the temporary datetime column to keep all original columns intact
    final_df = final_df.drop(columns=['LASTNIGHT_dt'])
    
    # Save the final subset
    final_df.to_csv(output_csv, index=False)
    print(f"\nSaved final subset to {output_csv}")
    print(f"Final count: {len(final_df)} observations across {final_df['TARGETID'].nunique()} unique targets.")
    if not final_df.empty:
        print(f"Highest chi2 in {output_csv}: {chi2.loc[final_df.index].max():.8g}")

    # 3. Create directories for plot sorting
    survived_dir = os.path.join("plots", "chi2_check", "survived")
    failed_dir = os.path.join("plots", "chi2_check", "failed")
    
    os.makedirs(survived_dir, exist_ok=True)
    os.makedirs(failed_dir, exist_ok=True)
    print("\nCopying PNGs to plot directories...")

    # Helper function to copy images based on a dataframe
    def copy_plots(dataframe, dest_folder):
        copied = 0
        missing = 0
        for tid, night in dataframe[["TARGETID", "LASTNIGHT"]].itertuples(index=False, name=None):
            if pd.isna(tid) or pd.isna(night):
                missing += 1
                continue
            # Match the YYYYMMDD filename for either CSV date format.
            night_str = night[:10].replace("-", "")
            
            png_name = f"recon_{tid}_{night_str}.png"
            src_path = os.path.join(recons_dir, tid, night_str, png_name)
            dest_path = os.path.join(dest_folder, png_name)
            
            if os.path.exists(src_path):
                # Using copy2 to preserve file metadata
                shutil.copy2(src_path, dest_path)
                copied += 1
            else:
                missing += 1
        return copied, missing

    #Copy the images that survived the chi2 cut
    s_copied, s_miss = copy_plots(survived_chi2_df, survived_dir)
    print(f"Survived Chi2 Cut: Copied {s_copied} images (Missing {s_miss} source files)")

    # Copy the images that failed the chi2 cut
    f_copied, f_miss = copy_plots(failed_chi2_df, failed_dir)
    print(f"Failed Chi2 Cut: Copied {f_copied} images (Missing {f_miss} source files)")

if __name__ == "__main__":
    process_catalogs_and_plots()