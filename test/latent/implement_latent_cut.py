import numpy as np
import pandas as pd

subset_target_path = "/work/10579/prisha/ls6/desi_project/final_subset.csv"
latent_all_path = "/work/10579/prisha/ls6/desi_project/latent_info.csv"

subset_target = pd.read_csv(subset_target_path, dtype={"TARGETID": "int64"})
latent_all = pd.read_csv(latent_all_path, dtype={"TARGETID": "int64"})

latent1_cut = -1.0
latent4_cut = -2.0

latent1_variance_cut = 0.5
latent_variance_cut = 0.5

def count():
    unique_targets_subset = set(subset_target["TARGETID"])
    unique_targets_latent = set(latent_all["TARGETID"])


    overlap = unique_targets_subset & unique_targets_latent
    missing = unique_targets_subset - unique_targets_latent

    print(f"unique targets in final_subset: {len(unique_targets_subset)}")
    print(f"unique targets in latent: {len(unique_targets_latent)}")
    print(f"targets in both: {len(overlap)}")
    print(f"final_subset targets with no latents: {len(missing)}")




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

def implement_latent1_4_cut(latent_subset, latent1_cut, latent4_cut):
    mask = (latent_subset["Latent1"] <= latent1_cut) & (latent_subset["Latent4"] <= latent4_cut)
    passing_coadds = latent_subset[mask]
    passed_ids = set(passing_coadds["TARGETID"])
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)]
    print(f"targets passing (at least one coadd Latent1 <= {latent1_cut} and Latent4 <= {latent4_cut} ): {len(passed_ids)}")
    print(f"coadds recorded for those targets: {len(passed_targets)}")
    
    out_path = f"/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/passed_targets_latent1_4_{latent1_cut}_{latent4_cut}.csv"
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    return passed_targets


def implement_latent1_variance_cut(latent_subset, latent1_variance_cut):
    latent1_std = latent1_std = latent_subset.groupby("TARGETID")["Latent1"].std(ddof=0) # Delta Degrees of Freedom N - 0
    latent1_variance = latent1_std**2
    passed_ids = set(latent1_variance.index[latent1_variance >= latent1_variance_cut]) # group key is TARGETID, so index is TARGETID
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)]

    print(f"targets passing (latent1 variance <= {latent1_variance_cut}): {len(passed_ids)}")
    print(f"coadds recorded for those targets: {len(passed_targets)}")
    
    out_path = f"/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/passed_targets_latent1_variance_{latent1_variance_cut}.csv"
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    return passed_targets

def positive_negative_slope(latent_subset, latent1_cut):
    low  = latent_subset["Latent1"] <= latent1_cut  
    high = latent_subset["Latent1"] > 0
    has_low  = low.groupby(latent_subset["TARGETID"]).any()
    has_high = high.groupby(latent_subset["TARGETID"]).any()

    passed_ids = set(has_low.index[has_low & has_high])
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)]
    print(f"targets passing (latent1 low <= {latent1_cut} & latent1 high > 0): {len(passed_ids)}")
    out_path = f"/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/passed_targets_latent1_low_{latent1_cut}_and_positive.csv"
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}")


        
def compute_variance_latent(latent_subset, latent_variance_cut, n_latents=10):
    latent_cols = [f"Latent{i}" for i in range(1, n_latents + 1)]

    all_var = latent_subset.groupby("TARGETID")[latent_cols].var(ddof=0)

    passed = {}
    for col in latent_cols:
        var = all_var[col]
        row_var = latent_subset["TARGETID"].map(var)
        passed_targets = latent_subset[row_var >= latent_variance_cut].copy()
        passed_targets[f"{col}_variance"] = row_var[row_var >= latent_variance_cut]

        n_targets = (var >= latent_variance_cut).sum()

        print(f"{col}: targets passing (variance >= {latent_variance_cut}): {n_targets}")
        print(f"{col}: coadds recorded for those targets: {len(passed_targets)}")

        out_path = (
            f"/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/"
            f"passed_targets_{col}_variance_{latent_variance_cut}.csv"
        )
        passed_targets.to_csv(out_path, index=False)
        print(f"saved {out_path}")

        passed[col] = passed_targets

    return passed 



def main():
    count()
    latent_subset = extract_subset_latents()
    implement_latent1_cut(latent_subset, latent1_cut)
    passed_targets = implement_latent1_4_cut(latent_subset, latent1_cut, latent4_cut)
    implement_latent1_variance_cut(latent_subset, latent1_variance_cut)
    positive_negative_slope(latent_subset, latent1_cut)
    compute_variance_latent(latent_subset, latent_variance_cut)


if __name__ == "__main__":
    main()