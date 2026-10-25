"""
goal: how can we only select score 5 with latent values? what matrics can we apply?

So goal of this script is to see what characteristics latent values of score 5 have.

1. Get all targets with score 5
2. Get latent values of these targets
3. mean or mean of each pair, higest value,  etc 
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_VISUAL_XLSX = Path( "/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/overlap/CLQ_visual.xlsx" )
LATENT_ALL = Path("/work/11161/kanyuni/ls6/quassiQ_project/latent/latent_all_targets.csv")


def extract(visual_inspection):
    results_5 = []
    results_1 = []
    for name, df in visual_inspection.items():
        df.columns = df.columns.str.strip()
        if "average" not in df.columns or "TARGETID" not in df.columns: # skip sheets without average column
            print(f"Skipping '{name}', columns are: {list(df.columns)}")
            continue
        bad_ids = ["39627673550455521", "39627741670147785", "39627823480046613", "39627902941135614", "39627927393931075", "39627963779521305", "39628358258002822", "39628011250649588", "39628518056791749", "39627872800870492"] # drop anomalies idk why 39627872800870492 was classified as 5
        df = df[~df["TARGETID"].isin(bad_ids)]

        df["average"] = pd.to_numeric(df["average"], errors="coerce") # text to numerical values
        score5 = df[df["average"] == 5][["TARGETID"]].assign(tab=name) # assign() adds another column of tabs with sheet name
        score1 = df[df["average"] == 1][["TARGETID"]].assign(tab=name) 
        results_5.append(score5)
        results_1.append(score1)

    out_5 = pd.concat(results_5, ignore_index=True) # glue them together
    out_1 = pd.concat(results_1, ignore_index=True)


    latent_csv = pd.read_csv(LATENT_ALL, dtype={"TARGETID": str})
    latent_csv.columns = latent_csv.columns.str.strip()
    score5_ids = out_5["TARGETID"].unique()          # unique IDs, in case a target appears in several tabs
    score1_ids = out_1["TARGETID"].unique()
    score5_latent = latent_csv[latent_csv["TARGETID"].isin(score5_ids)]
    score1_latent = latent_csv[latent_csv["TARGETID"].isin(score1_ids)]
    
    print(score5_ids) # this is not unique because there are still multiple epochs at this point
    print(score5_latent)

    score5_latent.to_csv("/work/11161/kanyuni/ls6/quassiQ_project/latent/revisit_newpipeline/score5_latent.csv", index=False)# highest score
    score1_latent.to_csv("/work/11161/kanyuni/ls6/quassiQ_project/latent/revisit_newpipeline/score1_latent.csv", index=False)# highest score

def main():
    visual_inspection = pd.read_excel(DEFAULT_VISUAL_XLSX, sheet_name=None, dtype={"TARGETID": str})
    extract(visual_inspection)

if __name__ == "__main__":
    main()