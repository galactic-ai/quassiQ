import numpy as np
import pandas as pd

subset_target_path = "/work/10579/prisha/ls6/desi_project/final_subset.csv"
latent_all_path = "/work/10579/prisha/ls6/desi_project/latent_info.csv"

subset_target = pd.read_csv(subset_target_path, dtype={"TARGETID": "int64"})
latent_all = pd.read_csv(latent_all_path, dtype={"TARGETID": "int64"})

latent1_cut = -1.0
latent4_cut = -2.0

latent1_variance_cut = 0.5
latent_variance_cut = 0.3

min_latents = 3
latent4_cut = -2.5


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



def implement_latent_i_variance_cut(latent_subset, i, latent_variance_cut):
    col = f"Latent{i}"
    variance = latent_subset.groupby("TARGETID")[col].var(ddof=0)

    passed_ids = set(variance.index[variance > latent_variance_cut])
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)].copy()
    passed_targets[f"{col}_variance"] = passed_targets["TARGETID"].map(variance)

    print(f"{col}: targets passing (variance > {latent_variance_cut}): {len(passed_ids)}")
    print(f"{col}: coadds recorded for those targets: {len(passed_targets)}")

    out_path = (
        f"/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/"
        f"passed_targets_{col}_variance_{latent_variance_cut}.csv"
    )
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    return passed_targets


def implement_latent4_cut(latent_subset, latent4_cut):
    passing_coadds = latent_subset[latent_subset["Latent4"] < latent4_cut]
    passed_ids = set(passing_coadds["TARGETID"])
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)]

    print(f"targets passing (at least one coadd Latent4 < {latent4_cut}): {len(passed_ids)}")
    print(f"coadds recorded for those targets: {len(passed_targets)}")

    out_path = f"/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/passed_targets_latent4_{latent4_cut}.csv"
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    return passed_targets

def implement_latent4_variance_and_cut(latent_subset, latent4_variance_cut, latent4_low_cut):
    variance = latent_subset.groupby("TARGETID")["Latent4"].var(ddof=0)

    # condition 1: variance of Latent4 >= cut
    var_ids = set(variance.index[variance >= latent4_variance_cut])

    # condition 2: at least one coadd with Latent4 < low cut
    low_ids = set(latent_subset.loc[latent_subset["Latent4"] < latent4_low_cut, "TARGETID"])

    passed_ids = var_ids & low_ids
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)].copy()
    passed_targets["Latent4_variance"] = passed_targets["TARGETID"].map(variance)

    print(f"targets with Latent4 variance >= {latent4_variance_cut}: {len(var_ids)}")
    print(f"targets with at least one coadd Latent4 < {latent4_low_cut}: {len(low_ids)}")
    print(f"targets passing both: {len(passed_ids)}")
    print(f"coadds recorded for those targets: {len(passed_targets)}")

    out_path = (
        f"/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/"
        f"passed_targets_latent4_variance_{latent4_variance_cut}_and_low_{latent4_low_cut}.csv"
    )
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    return passed_targets

def implement_latent1_neg_latent8_pos_cut(
    latent_subset,
    variance_cut=0.5,
    latent1_low_cut=-1.8,
    latent8_high_cut=1.8,
):
    var1 = latent_subset.groupby("TARGETID")["Latent1"].var(ddof=0)
    var8 = latent_subset.groupby("TARGETID")["Latent8"].var(ddof=0)

    # variance conditions
    var1_ids = set(var1.index[var1 >= variance_cut])
    var8_ids = set(var8.index[var8 >= variance_cut])

    # at least one coadd Latent1 negative, at least one coadd Latent8 positive
    low1_ids = set(latent_subset.loc[latent_subset["Latent1"] < latent1_low_cut, "TARGETID"])
    high8_ids = set(latent_subset.loc[latent_subset["Latent8"] > latent8_high_cut, "TARGETID"])

    ids1 = var1_ids & low1_ids
    ids8 = var8_ids & high8_ids
    passed_ids = ids1 & ids8

    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)].copy()
    passed_targets["Latent1_variance"] = passed_targets["TARGETID"].map(var1)
    passed_targets["Latent8_variance"] = passed_targets["TARGETID"].map(var8)

    print(f"Latent1: variance >= {variance_cut} and at least one coadd < {latent1_low_cut}: {len(ids1)}")
    print(f"Latent8: variance >= {variance_cut} and at least one coadd > {latent8_high_cut}: {len(ids8)}")
    print(f"targets passing both: {len(passed_ids)}")
    print(f"coadds recorded for those targets: {len(passed_targets)}")

    out_path = (
        f"/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/"
        f"passed_targets_latent1_neg{latent1_low_cut}_latent8_pos{latent8_high_cut}_variance_{variance_cut}.csv"
    )
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    return passed_targets


def implement_latent1_8_variance_cut(
    latent_subset,
    variance_cut=0.5,
    latent1_low_cut=None,   # e.g. -1.8 -> need at least one coadd Latent1 < cut; None = skip
    latent8_high_cut=None,  # e.g. 1.8  -> need at least one coadd Latent8 > cut; None = skip
):
    var1 = latent_subset.groupby("TARGETID")["Latent1"].var(ddof=0)
    var8 = latent_subset.groupby("TARGETID")["Latent8"].var(ddof=0)

    ids1 = set(var1.index[var1 > variance_cut])
    ids8 = set(var8.index[var8 > variance_cut])
    print(f"Latent1 variance > {variance_cut}: {len(ids1)}")
    print(f"Latent8 variance > {variance_cut}: {len(ids8)}")

    if latent1_low_cut is not None:
        low1 = set(latent_subset.loc[latent_subset["Latent1"] < latent1_low_cut, "TARGETID"])
        ids1 &= low1
        print(f"Latent1 also with at least one coadd < {latent1_low_cut}: {len(ids1)}")

    if latent8_high_cut is not None:
        high8 = set(latent_subset.loc[latent_subset["Latent8"] > latent8_high_cut, "TARGETID"])
        ids8 &= high8
        print(f"Latent8 also with at least one coadd > {latent8_high_cut}: {len(ids8)}")

    passed_ids = ids1 & ids8
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)].copy()
    passed_targets["Latent1_variance"] = passed_targets["TARGETID"].map(var1)
    passed_targets["Latent8_variance"] = passed_targets["TARGETID"].map(var8)

    print(f"targets passing both: {len(passed_ids)}")
    print(f"coadds recorded for those targets: {len(passed_targets)}")

    tag = f"variance_{variance_cut}"
    if latent1_low_cut is not None:
        tag += f"_latent1_neg{latent1_low_cut}"
    if latent8_high_cut is not None:
        tag += f"_latent8_pos{latent8_high_cut}"
    out_path = f"/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/passed_targets_latent1_8_{tag}.csv"
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    return passed_targets

import pandas as pd

path_template = "/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/passed_targets_Latent{i}_variance_0.5.csv"

manual_targets = [
    39627868602371358,
    39627879578861949,
    39633100518459964,
    39627963825660206,
    39628313819355032,
    39628502256844881,
    39632946436505945,
]


def check_overlap(manual_targets):
    manual_set = set(manual_targets)
    ids_per_latent = {}

    # load the target ids from each file
    for i in range(1, 11):
        path = path_template.format(i=i)
        try:
            df = pd.read_csv(path, dtype={"TARGETID": "int64"})
        except FileNotFoundError:
            print(f"Latent{i}: file not found, skipping")
            continue
        ids_per_latent[i] = set(df["TARGETID"])

    # 1) overlap of each file with your typed list
    print("--- overlap with manual list ---")
    for i, ids in ids_per_latent.items():
        overlap = ids & manual_set
        print(f"Latent{i}: {len(ids)} targets in file, {len(overlap)} in manual list")

    # 2) overlap between the files themselves
    print()
    print("--- overlap between files ---")
    all_sets = list(ids_per_latent.values())
    in_all = set.intersection(*all_sets)
    in_any = set.union(*all_sets)
    print(f"targets in every file: {len(in_all)}")
    print(f"targets in at least one file: {len(in_any)}")

    # 3) for each manual target, how many files contain it
    print()
    print("--- manual targets: how many files contain each ---")
    for t in manual_targets:
        files = [i for i, ids in ids_per_latent.items() if t in ids]
        print(f"{t}: in {len(files)} files {files}")

def detect_high_variance_targets(latent_subset, latents=(1, 4, 6, 8), var_cut=0.3):
    cols = [f"Latent{i}" for i in latents]

    # variance of each latent across a target's coadds
    var_df = latent_subset.groupby("TARGETID")[cols].var()

    # True only if every one of the latents is above the cut
    passed = (var_df > var_cut).all(axis=1)
    passed_ids = set(var_df.index[passed])

    # keep ALL coadds of the passing targets
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)]

    print(f"targets with variance > {var_cut} in all of {cols}: {len(passed_ids)}")
    print(f"coadds recorded for those targets: {len(passed_targets)}")

    out_path = (
        "/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/"
        f"passed_targets_latent{'_'.join(map(str, latents))}_variance_{var_cut}_all.csv"
    )
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    return passed_targets

def detect_min_latents_high_variance(latent_subset, latents=(1, 4, 5, 6, 8), var_cut=0.5, min_count=2):
    cols = [f"Latent{i}" for i in latents]

    # variance of each latent across a target's coadds
    var_df = latent_subset.groupby("TARGETID")[cols].var()

    # how many of the latents exceed the cut, per target
    n_high = (var_df > var_cut).sum(axis=1)
    passed_ids = set(var_df.index[n_high >= min_count])

    # keep ALL coadds of the passing targets
    passed_targets = latent_subset[latent_subset["TARGETID"].isin(passed_ids)]

    print(f"targets with at least {min_count} of {cols} having variance > {var_cut}: {len(passed_ids)}")
    print(f"coadds recorded for those targets: {len(passed_targets)}")

    out_path = (
        "/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/"
        f"passed_targets_latent{'_'.join(map(str, latents))}_variance_{var_cut}_atleast{min_count}.csv"
    )
    passed_targets.to_csv(out_path, index=False)
    print(f"saved {out_path}")

    return passed_targets

def main():
    count()
    latent_subset = extract_subset_latents()
    # implement_latent1_cut(latent_subset, latent1_cut)
    # passed_targets = implement_latent1_4_cut(latent_subset, latent1_cut, latent4_cut)
    # implement_latent1_variance_cut(latent_subset, latent1_variance_cut)
    # positive_negative_slope(latent_subset, latent1_cut)
    compute_variance_latent(latent_subset, latent_variance_cut)
    #mplement_latent_i_variance_cut(latent_subset, 4, 0.7) 
    # #implement_latent4_cut(latent_subset, latent4_cut)
    #implement_latent4_variance_and_cut(latent_subset, 0.5, 0)
    # #implement_latent1_neg_latent8_pos_cut(latent_subset, 0.5, 0, 1.0)
    # implement_latent1_8_variance_cut(latent_subset, 0.5)
    # implement_latent1_8_variance_cut(latent_subset, 0.5, latent1_low_cut=-1.0, latent8_high_cut=1.0)
    check_overlap(manual_targets)
    #passed = detect_high_variance_targets(latent_subset)
    passed = detect_min_latents_high_variance(latent_subset)

    
if __name__ == "__main__":
    main()