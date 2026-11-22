"""Correlation between SNR and the VI scores (my scores, and CLQ-in-normalized scores).

latent_info.csv has no SNR, so SNR is taken from final_subset.csv (column MEDIAN_SNR,
one value per coadd). Each target gets the median and the minimum of MEDIAN_SNR over its
coadds. Scores come from latent_VI_pies.py (same rules: lowest score for repeated targets;
dark yellow = 5, light yellow = 4 for CLQ in normalized spectra).

Usage:
    python /work/11161/kanyuni/ls6/quassiQ_project/quassiQ/test/latent/analyze/snr_latent_score.py --xlsx /work/11161/kanyuni/ls6/quassiQ_project/vi/score/latent_VI.xlsx 
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import latent_vi_pi as vi

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def snr_per_target(subset_path):
    s = pd.read_csv(subset_path, usecols=["TARGETID", "MEDIAN_SNR"], dtype={"TARGETID": "int64"})
    g = s.groupby("TARGETID")["MEDIAN_SNR"]
    return pd.DataFrame({"snr_median": g.median(), "snr_min": g.min(), "n_coadds": g.size()})


def corr_row(label, version, x, y):
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    rho, p = spearmanr(x, y)
    r, pr = pearsonr(x, y)
    return {"sample": label, "scores": version, "n": int(ok.sum()),
            "spearman_rho": round(rho, 3), "spearman_p": round(p, 4),
            "pearson_r": round(r, 3), "pearson_p": round(pr, 4)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default="latent_VI.xlsx")
    p.add_argument("--subset", default="/work/10579/prisha/ls6/desi_project/final_subset.csv")
    p.add_argument("--out_dir", default=".")
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    df = vi.add_scores(vi.read_workbook(a.xlsx))
    snr = snr_per_target(a.subset)

    # one row per target over all six sheets: my scores and CLQ-in-normalized scores
    mine, clq = vi.combined_scores(df, [int(s) for s in vi.LATENT_SHEETS])
    tgt = pd.DataFrame({"score": mine, "score_clq": clq}).join(snr)
    print(f"targets with a score: {len(tgt)}; with SNR: {int(tgt.snr_median.notna().sum())}")
    tgt.to_csv(os.path.join(a.out_dir, "vi_scores_snr_per_target.csv"))

    rows = []
    for snr_col, snr_name in [("snr_median", "median SNR"), ("snr_min", "min SNR")]:
        for col, version in [("score", "scores"), ("score_clq", "CLQ in normalized")]:
            rows.append(corr_row(f"all targets, {snr_name}", version, tgt[snr_col], tgt[col]))
    # per latent sheet (a target on several sheets appears on each of them)
    d = df.join(snr, on="TARGETID")
    for latent in [int(s) for s in vi.LATENT_SHEETS]:
        sub = d[d["latent"] == latent]
        for col, version in [("score_lowest", "scores"), ("score_clq", "CLQ in normalized")]:
            rows.append(corr_row(f"Latent {latent}, median SNR", version, sub["snr_median"], sub[col]))
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(a.out_dir, "vi_snr_correlation.csv"), index=False)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))

    # figure: median SNR for each score, one panel per version
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, col, version in zip(axes, ["score", "score_clq"], ["Scores", "CLQ in normalized"]):
        t = tgt.dropna(subset=["snr_median", col])
        levels = sorted(t[col].unique())
        data = [t.loc[t[col] == lv, "snr_median"].values for lv in levels]
        ax.boxplot(data, positions=range(len(levels)), widths=0.55, showfliers=False,
                   medianprops={"color": "black"})
        for i, vals in enumerate(data):
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=22, alpha=0.7,
                       color="tab:blue", zorder=3)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([f"{lv:g}\n(n={len(v)})" for lv, v in zip(levels, data)])
        rho, pv = spearmanr(t["snr_median"], t[col])
        ax.set_title(f"{version}\nSpearman rho = {rho:.2f}, p = {pv:.2g}, n = {len(t)}")
        ax.set_xlabel("VI score")
    axes[0].set_ylabel("Median SNR per target")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out_dir, "snr_vs_score.png"), dpi=200)


if __name__ == "__main__":
    main()