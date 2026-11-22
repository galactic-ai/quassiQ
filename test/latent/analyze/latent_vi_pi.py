"""
python /work/11161/kanyuni/ls6/quassiQ_project/quassiQ/test/latent/analyze/latent_vi_pi.py  --xlsx /work/11161/kanyuni/ls6/quassiQ_project/vi/score/latent_VI.xlsx  --combine 4 6
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import pandas as pd

LATENT_SHEETS = ["1", "3", "4", "5", "6", "7"]
DARK_YELLOW = "FFFFE599"    # -> score 5
LIGHT_YELLOW = "FFFFF2CC"   # -> score 4
BINS = [("[1, 3)", 1, 3), ("[3, 4)", 3, 4), ("[4, 5]", 4, 5.0001)]
COLORS = ["tab:blue", "tab:orange", "tab:green"]      # [1, 3), [3, 4), [4, 5]

# font: matplotlib's default (DejaVu Sans)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


def target_id(v):
    """TARGETIDs are sometimes stored as floats (e.g. 3.96279575761417e+16)."""
    return int(round(float(v))) if isinstance(v, (int, float)) else int(str(v).strip())


def read_workbook(path):
    wb = openpyxl.load_workbook(path)
    rows = []
    for name in LATENT_SHEETS:
        for r in wb[name].iter_rows(min_col=1, max_col=2):
            if r[0].value is None:
                continue
            fill = r[0].fill
            rgb = fill.fgColor.rgb if (fill and fill.fill_type and fill.fgColor.type == "rgb") else None
            rows.append({
                "latent": int(name),
                "TARGETID": target_id(r[0].value),
                "score": r[1].value,
                "yellow": "dark" if rgb == DARK_YELLOW else "light" if rgb == LIGHT_YELLOW else "none",
            })
    return pd.DataFrame(rows)


def add_scores(df):
    # lowest score per target across all sheets (blank scores are ignored)
    lowest = df.dropna(subset=["score"]).groupby("TARGETID")["score"].min()
    df = df.copy()
    df["score_lowest"] = df["TARGETID"].map(lowest)
    # CLQ in normalized spectra: dark yellow -> 5, light yellow -> 4
    df["score_clq"] = df["score_lowest"]
    df.loc[df["yellow"] == "dark", "score_clq"] = 5.0
    df.loc[df["yellow"] == "light", "score_clq"] = 4.0
    return df


def bin_counts(scores):
    scores = scores.dropna()
    return [int(((scores >= lo) & (scores < hi)).sum()) for _, lo, hi in BINS]


def combined_scores(df, latents):
    """One score per target over the given sheets: (Scores, CLQ-in-normalized scores)."""
    sub = df[df["latent"].isin(latents)]
    per_target = sub.groupby("TARGETID").agg(
        lowest=("score_lowest", "first"),
        any_dark=("yellow", lambda s: (s == "dark").any()),
        any_light=("yellow", lambda s: (s == "light").any()),
    )
    clq = per_target["lowest"].copy()
    clq[per_target["any_dark"]] = 5.0
    clq[per_target["any_light"]] = 4.0      # a light highlight anywhere -> the lower value, 4
    return per_target["lowest"], clq


def pie_on_axis(ax, counts, title):
    total = sum(counts)
    wedges, _ = ax.pie(counts, colors=COLORS, startangle=90, counterclock=False,
                       wedgeprops={"edgecolor": "white", "linewidth": 1})
    for w, n in zip(wedges, counts):
        if n == 0:
            continue
        pct = 100 * n / total
        ang = np.deg2rad((w.theta1 + w.theta2) / 2)
        inside = pct >= 10                      # small slices get their label outside
        r = 0.62 if inside else 1.18
        ax.text(r * np.cos(ang), r * np.sin(ang), f"{pct:.1f}%\n(n={n})",
                ha="center", va="center", fontsize=10)
    ax.set_title(f"{title}  (n={total})", fontsize=13, pad=26)


def legend(fig):
    handles = [plt.matplotlib.patches.Patch(color=c, label=b[0]) for c, b in zip(COLORS, BINS)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=12, frameon=False, title="Score")


def draw_per_latent(df, col, title, out_path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    for ax, latent in zip(axes.ravel(), [int(s) for s in LATENT_SHEETS]):
        pie_on_axis(ax, bin_counts(df.loc[df["latent"] == latent, col]), f"Latent {latent}")
    legend(fig)
    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def draw_combined(df, latents, out_path):
    mine, clq = combined_scores(df, latents)
    name = ", ".join(str(l) for l in latents)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    pie_on_axis(axes[0], bin_counts(mine), "Scores")
    pie_on_axis(axes[1], bin_counts(clq), "CLQ in normalized)")
    legend(fig)
    fig.suptitle(f"VI score breakdown, latents {name} combined (each target counted once)", fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return bin_counts(mine), bin_counts(clq)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xlsx", default="latent_VI.xlsx")
    p.add_argument("--out_dir", default=".")
    p.add_argument("--combine", type=int, nargs="+", default=[4, 5, 6, 7])
    a = p.parse_args()

    df = add_scores(read_workbook(a.xlsx))
    os.makedirs(a.out_dir, exist_ok=True)

    for col, name, title in [
        ("score_lowest", "score_breakdown_latents.png",
         "VI score breakdown per latent (lowest score kept for repeated targets)"),
        ("score_clq", "score_breakdown_latents_clq_normalized.png",
         "VI score breakdown per latent with CLQ in normalized spectra "
         "(dark yellow = 5, light yellow = 4)"),
    ]:
        draw_per_latent(df, col, title, os.path.join(a.out_dir, name))
        print(name)
        for latent in [int(s) for s in LATENT_SHEETS]:
            c = bin_counts(df.loc[df["latent"] == latent, col])
            print(f"  Latent {latent}: " + ", ".join(f"{b[0]}={n}" for b, n in zip(BINS, c)) + f"  (total {sum(c)})")

    tag = "".join(str(l) for l in a.combine)
    out = os.path.join(a.out_dir, f"score_breakdown_latents_{tag}_combined.png")
    mine, clq = draw_combined(df, a.combine, out)
    print(os.path.basename(out))
    print("  my VI scores: " + ", ".join(f"{b[0]}={n}" for b, n in zip(BINS, mine)) + f"  (total {sum(mine)})")
    print("  CLQ in normalized: " + ", ".join(f"{b[0]}={n}" for b, n in zip(BINS, clq)) + f"  (total {sum(clq)})")


if __name__ == "__main__":
    main()