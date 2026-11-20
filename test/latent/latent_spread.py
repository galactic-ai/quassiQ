"""Targets (from final_subset) whose spread (max - min across epochs) in one latent exceeds a cut."""
import argparse
import os
import pandas as pd

SUBSET_PATH = "/work/10579/prisha/ls6/desi_project/final_subset.csv"   
LATENT_PATH = "/work/10579/prisha/ls6/desi_project/latent_info.csv"   
OUT_DIR = "/work/11161/kanyuni/ls6/quassiQ_project/latent/passed/spread/single"

p = argparse.ArgumentParser()
p.add_argument("--latent", type=int, default=6)
p.add_argument("--spread_cut", type=float, default=1.5)
args = p.parse_args()
col = f"Latent{args.latent}"

# targets come from final_subset, latent values come from latent_info
subset_ids = set(pd.read_csv(SUBSET_PATH, usecols=["TARGETID"], dtype={"TARGETID": "int64"})["TARGETID"])
latent = pd.read_csv(LATENT_PATH, dtype={"TARGETID": "int64"})
latent = latent[latent["TARGETID"].isin(subset_ids)]

# spread of the latent across each target's epochs
g = latent.groupby("TARGETID")[col]
table = pd.DataFrame({"n_obs": g.size(), "min": g.min(), "max": g.max()})
table["spread"] = table["max"] - table["min"]
table = table[table["n_obs"] >= 2]
passed = table[table["spread"] > args.spread_cut].sort_values("spread", ascending=False).round(3)

print(f"{col}: spread > {args.spread_cut}")
print(f"targets in final_subset: {len(subset_ids)}, checked (>= 2 epochs): {len(table)}, passed: {len(passed)}")

os.makedirs(OUT_DIR, exist_ok=True)
out_path = os.path.join(OUT_DIR, f"passed_targets_latent{args.latent}_spread_above_{args.spread_cut:g}.csv")
passed.to_csv(out_path)
print(f"saved {out_path}")