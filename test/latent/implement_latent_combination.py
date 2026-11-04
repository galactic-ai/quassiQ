# explore which combinations is the best 

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from itertools import combinations
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

    
"""
    Latent3  Latent7  Latent9
    TARGETID
    1001           True     True    False
    1002          False     True     True

for one target
    Latent3     True
    Latent7     True
    Latent9    False

 """
def detect_min_latents_high_variance(latent_subset, combos, var_cut=0.5, min_count=2):
    cols = [f"Latent{i}" for i in combos]

    # variance of each latent across a target's coadds
    var_df = latent_subset.groupby("TARGETID")[cols].var()

    # how many of the latents exceed the cut, per target
    n_high = (var_df > var_cut).sum(axis=1)
    passed_ids = set(var_df.index[n_high >= min_count])
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)].copy()
    
    high_mask = var_df > var_cut
    high_pairs = high_mask.apply(
        lambda row: ";".join(
            f"{a}&{b}" for a, b in combinations([c for c in cols if row[c]], 2)
        ),
        axis=1,
    )
    passed_targets["high_var_pairs"] = passed_targets["TARGETID"].map(high_pairs)
    # keep ALL coadds of the passing targets
    

    print(f"targets with at least {min_count} of {cols} having variance > {var_cut}: {len(passed_ids)}")
    print(f"coadds recorded for those targets: {len(passed_targets)}")

    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f"passed_targets_latent{'_'.join(map(str, combos))}_variance_{var_cut}_atleast{min_count}.csv",)
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}\n")
    return passed_targets



def plot_top_pairs(passed_targets, combo, top_n=15, out_dir="/work/11161/kanyuni/ls6/quassiQ_project/latent/latent_combination/frequent_pairs", fname="top_pairs.png"):
    # one row per target, so each target is counted once
    per_target = passed_targets.drop_duplicates("TARGETID")

    # "A&B;A&C" -> one pair per row, then count
    pair_counts = (
        per_target["high_var_pairs"]
        .dropna()
        .loc[lambda s: s != ""]        # skip targets with no pairs
        .str.split(";")
        .explode()
        .value_counts()
    )

    print(pair_counts.head(top_n))

    top = pair_counts.head(top_n)

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 2))
    ax.barh(top.index[::-1], top.values[::-1])   # reversed so the biggest is on top
    ax.set_xlabel("Number of targets")
    ax.set_title(f"Top {len(top)} latent pairs with variance above cut\nlatent: {combo}")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)   
    out_path = os.path.join(out_dir, fname)  
    fig.savefig(out_path, dpi=200)   
    print(f'saved {out_path}')

    plt.show()
    return pair_counts

def main():
    latent_subset = extract_subset_latents()
    results = {}

    for latent in combos:
        print(f"combination {latent}")
        passed = detect_min_latents_high_variance(
            latent_subset, latent, var_cut, min_count
        )
        results[latent] = passed

        tag = "_".join(map(str, latent))
        plot_top_pairs(
            passed,
            latent,
            top_n=15,
            fname=f"top_pairs_latent{tag}.png", 
            
        )

    return results
    
if __name__ == "__main__":
    main()