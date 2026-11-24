"""Targets (from final_subset) whose spread (max - min across epochs) in one latent exceeds a cut."""
import argparse
import os
import pandas as pd
import random

SUBSET_PATH = "/work/10579/prisha/ls6/desi_project/final_subset.csv"   
LATENT_PATH = "/work/10579/prisha/ls6/desi_project/latent_info.csv"   
OUT_DIR = "/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/spread/single"

p = argparse.ArgumentParser()
p.add_argument("--latent", type=int, nargs="+", default=[1, 4, 5, 6, 7])
p.add_argument("--spread_cut", type=float, default=1.0)
args = p.parse_args()
cols = [f"Latent{i}" for i in args.latent]
# targets come from final_subset, latent values come from latent_info
subset_ids = set(pd.read_csv(SUBSET_PATH, usecols=["TARGETID"], dtype={"TARGETID": "int64"})["TARGETID"])
latent = pd.read_csv(LATENT_PATH, dtype={"TARGETID": "int64"})
latent = latent[latent["TARGETID"].isin(subset_ids)]

# spread of the latent across each target's epochs
g = latent.groupby("TARGETID")[cols]
spreads = g.max() - g.min()
spreads = spreads.loc[g.size() >= 2] # 103, 101 etc
passed = spreads.loc[spreads.gt(args.spread_cut).any(axis=1)]
sampled_ids = random.sample(passed.index.tolist(), k=200)
sampled = passed.loc[sampled_ids]


print(f"{len(passed)} targets passed; sampled {len(sampled)}")
for target_id in sampled_ids:
    print(target_id)
os.makedirs(OUT_DIR, exist_ok=True)
latent_label = "_".join(map(str, args.latent)) # map: apply to each item
out_path = os.path.join(
    OUT_DIR, f"sampled_200_latents_{latent_label}_spread_above_{args.spread_cut:g}.csv"
)
sampled.round(3).to_csv(out_path)
print(f"Saved {out_path}")