# explore which combinations is the best 

import numpy as np
import pandas as pd
import os

subset_target_path = "/work/10579/prisha/ls6/desi_project/final_subset.csv"
latent_all_path = "/work/10579/prisha/ls6/desi_project/latent_info.csv"

subset_target = pd.read_csv(subset_target_path, dtype={"TARGETID": "int64"})
latent_all = pd.read_csv(latent_all_path, dtype={"TARGETID": "int64"})
out_dir = "/work/11161/kanyuni/ls6/quassiQ_project/latent/latent_combination"

combos = [(1, 4, 5, 6, 8), # best?
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), # all
            (1, 3, 4, 5, 6, 8, 9, 10)]  # exclude 2 and 7 and include 3 and 9, 10 
var_cut = 0.5
min_count = 2

def extract_subset_latents():
    subset_ids = set(subset_target["TARGETID"])
    latent_subset = latent_all[latent_all["TARGETID"].isin(subset_ids)]

    print(f"coadds kept: {len(latent_subset)}")
    print(f"unique targets found: {latent_subset['TARGETID'].nunique()}")

    return latent_subset

def detect_min_latents_high_variance(latent_subset, combos, var_cut=0.5, min_count=2):
    cols = [f"Latent{i}" for i in combos]

    # variance of each latent across a target's coadds
    var_df = latent_subset.groupby("TARGETID")[cols].var()

    # how many of the latents exceed the cut, per target
    n_high = (var_df > var_cut).sum(axis=1)
    passed_ids = set(var_df.index[n_high >= min_count])

    # keep ALL coadds of the passing targets
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)]

    print(f"targets with at least {min_count} of {cols} having variance > {var_cut}: {len(passed_ids)}")
    print(f"coadds recorded for those targets: {len(passed_targets)}")

    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f"passed_targets_latent{'_'.join(map(str, combos))}_variance_{var_cut}_atleast{min_count}.csv",)
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}\n")
    return passed_targets

def main():
    latent_subset = extract_subset_latents()
    results = {}
    for latent in combos:
        print(f"combination {latent} ")
        results[latent] = detect_min_latents_high_variance(
          latent_subset, latent, var_cut, min_count
        )
    return results
    
if __name__ == "__main__":
    main()