import numpy as np
import pandas as pd

subset_target_path = "/work/10579/prisha/ls6/desi_project/final_subset.csv"
latent_all_path = "/work/11161/kanyuni/ls6/quassiQ_project/latent/latent_all_targets.csv"

subset_target = pd.read_csv(latent_all_path, dtype={"TARGETID": "int64"})
latent_all = pd.read_csv(latent_all_path, dtype={"TARGETID": "int64"})

latent1_cut = -1.8

def count():
    print(f"unique targets in final_subset: {len(set(subset_target['TARGETID']))}")


def extract_subset_latents():
    subset_ids = set(subset_target["TARGETID"])
    latent_subset = latent_all[latent_all["TARGETID"].isin(subset_ids)]

    print(f"coadds kept: {len(latent_subset)}")
    print(f"unique targets found: {len(set(latent_subset['TARGETID']))}")

    return latent_subset

def implement_latent1_cut(latent_subset, latent1_cut):
    passing_coadds = latent_subset[latent_subset["Latent1"] <= latent1_cut]
    passed_ids = set(passing_coadds["TARGETID"])
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)]
    print(f"targets passing (at least one coadd Latent1 <= {latent1_cut}): {len(passed_ids)}")
    print(f"coadds recorded for those targets: {len(passed_targets)}")
    
    out_path = f"/work/11161/kanyuni/ls6/quassiQ_project/latent/passed_targets_latent1_{latent1_cut}.csv"
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    return passed_targets



    

def main():
    count()
    latent_subset = extract_subset_latents()
    passed_targets = implement_latent1_cut(latent_subset, latent1_cut)

if __name__ == "__main__":
    main()