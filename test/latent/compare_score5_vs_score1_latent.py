from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv = {
"score5" : "/work/11161/kanyuni/ls6/quassiQ_project/latent/revisit_newpipeline/score5_latent.csv",
"score1" : "/work/11161/kanyuni/ls6/quassiQ_project/latent/revisit_newpipeline/score1_latent.csv"
}
latent_cols = [f"Latent{i}" for i in range (1,11)]

# mean of each latent, mean of absolute values 
# highest value of latent (same target) --> maybe one latent captures flux or somthing? then difference should be change for score 5 ones
def count():
    for name, path in csv.items():
        df = pd.read_csv(path, dtype={"TARGETID": "int64"})
        print(f"{name} target: {len(set(df["TARGETID"]))}")
        print(f"{name} coadd: {len(set(df["COADD_TAG"]))}")


def max_min():
    for name, path in csv.items():
        df = pd.read_csv(path, dtype={"TARGETID": "int64"})
        
        grouped = df.groupby("TARGETID")[latent_cols]
        diff = grouped.max() - grouped.min()

        diff.columns = [f"{c}_diff" for c in latent_cols]

        print(diff)
        diff.to_csv(f"/work/11161/kanyuni/ls6/quassiQ_project/latent/revisit_newpipeline/latent_differences_{name}.csv")
        print(f"saved latent_differences_{name}.csv")

def mean():
    for name, path in csv.items():
        df = pd.read_csv(path, dtype={"TARGETID": "int64"})
        
        grouped = df.groupby("TARGETID")[latent_cols]
        mean = grouped.mean()
        mean.columns = [f"{c}_mean" for c in latent_cols]

        print(mean)
        mean.to_csv(f"/work/11161/kanyuni/ls6/quassiQ_project/latent/revisit_newpipeline/latent_means_{name}.csv")  
        print(f"saved latent_means_{name}.csv")

def absolute_mean():
    for name, path in csv.items():
        df = pd.read_csv(path, dtype={"TARGETID": "int64"})
        
        abs_df = df[["TARGETID"] + latent_cols].copy()
        abs_df[latent_cols] = abs_df[latent_cols].abs()

        abs_mean_df = abs_df.groupby("TARGETID")[latent_cols].mean()
        abs_mean_df.columns = [f"{c}_abs_mean" for c in latent_cols]

        print(abs_mean_df)
        abs_mean_df.to_csv(f"/work/11161/kanyuni/ls6/quassiQ_project/latent/revisit_newpipeline/latent_absolute_means_{name}.csv")  
        print(f"saved latent_absolute_means_{name}.csv")

def variance():
    for name, path in csv.items():
        df = pd.read_csv(path, dtype={"TARGETID": "int64"})
        
        grouped = df.groupby("TARGETID")[latent_cols]
        variance = grouped.var()
        variance.columns = [f"{c}_var" for c in latent_cols]

        print(variance)
        variance.to_csv(f"/work/11161/kanyuni/ls6/quassiQ_project/latent/revisit_newpipeline/latent_variance_{name}.csv")  
        print(f"saved latent_variance_{name}.csv")


def absolute_variance():
    for name, path in csv.items():
        df = pd.read_csv(path, dtype={"TARGETID": "int64"})
        
        abs_df = df[["TARGETID"] + latent_cols].copy()
        abs_df[latent_cols] = abs_df[latent_cols].abs()

        abs_variance_df = abs_df.groupby("TARGETID")[latent_cols].var()
        abs_variance_df.columns = [f"{c}_abs_var" for c in latent_cols]

        print(abs_variance_df)
        abs_variance_df.to_csv(f"/work/11161/kanyuni/ls6/quassiQ_project/latent/revisit_newpipeline/latent_absolute_variance_{name}.csv") 
        print(f"saved latent_absolute_variance_{name}.csv")



def main():
    count()
    # max_min()
    # mean()
    # absolute_mean()
    # variance()
    # absolute_variance()

if __name__ == "__main__":
    main()
