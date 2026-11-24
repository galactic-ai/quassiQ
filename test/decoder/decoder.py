import os
import sys
sys.path.append("/work/11161/kanyuni/ls6/quassiQ_project/spenderq/src")
import argparse

import numpy as np
import torch
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from spenderq.spenderq import SpenderQ
import pandas as pd

latent_all = pd.read_csv("/work/10579/prisha/ls6/desi_project/latent_info.csv",
                         dtype={"TARGETID": "int64"})
cols = [f"Latent{i}" for i in range(1, 11)]

    
out_root = "/work/11161/kanyuni/ls6/quassiQ_project/decoder"


# highest, lowest, 16th, 84th, median
#setting upper and lower limit by eye
# NEW_VAL = {
#     1:  (-3.1142, -1.8213, -0.2580, 5.1101),
#     2:  (-1.9557,  0.0229,  1.3265, 4.2888),
#     3:  (-3.3126, -0.1790,  0.6263, 2.0338),
#     4:  (-2.8105, -1.4598,  0.0710, 5.2354),
#     5:  (-7.5185, -0.6372,  0.5901, 1.8542),
#     6:  (-4.2829, -0.9653,  0.1626, 1.3293),
#     7:  (-1.6595, -0.8453,  0.0745, 3.7149),
#     8:  (-4.2252, -0.0861,  2.0186, 7.6572),
#     9:  (-5.5418, -0.3044,  0.9916, 3.8828),
#     10: (-3.3234, -0.3705,  0.7154, 2.5936),
# }
# COLORS = ['tab:purple', 'tab:blue', 'tab:green', 'tab:orange']
# LABELS = ['min', '16th', '84th', 'max']

# np.linspace(np.min(), np.max(), 20)

parser = argparse.ArgumentParser()
parser.add_argument("--n_values", type=int, nargs="+", default=[5, 8, 10, 20])
args = parser.parse_args()

RANGES = {
    1:  (-3.1142, 5.1101),
    2:  (-1.9557, 4.2888),
    3:  (-3.3126, 2.0338),
    4:  (-2.8105, 5.2354),
    5:  (-7.5185, 1.8542),
    6:  (-4.2829, 1.3293),
    7:  (-1.6595, 3.7149),
    8:  (-4.2252, 7.6572),
    9:  (-5.5418, 3.8828),
    10: (-3.3234, 2.5936),
}

# N_SWEEP = 21 
# from corner plot (latent_corner.py)
P16 = np.array([-1.8213,  0.0229, -0.1790, -1.4598, -0.6372,
                -0.9653, -0.8453, -0.0861, -0.3044, -0.3705], dtype=np.float32)
MEDIAN = np.array([-1.2561,  0.6103,  0.2944, -0.8168,  0.1343,
                   -0.3024, -0.5056,  0.8830,  0.5331,  0.2229], dtype=np.float32)
latent_default = MEDIAN.copy()  
P84 = np.array([-0.2580,  1.3265,  0.6263,  0.0710,  0.5901,
                 0.1626,  0.0745,  2.0186,  0.9916,  0.7154], dtype=np.float32)

latent_default__39633100518459964 = np.array(   # 20220529
    [-0.8518244, -0.28353, 0.1625374, -0.0392417, 0.4274702,
     -0.4180194, -0.99107003, -0.49524423, 0.3647856, -0.34705868], dtype=np.float32)

latent_other__39633100518459964 = np.array(     # 20210519
    [-0.8557018, 0.49587187, 0.26037264, -0.17996126, -0.044506997,
     -0.6434802, -0.4695006, 0.35420156, 0.461946, 0.05186852], dtype=np.float32)

OFFSETS = np.array([-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2])

# 16th, median, 84th only

#NEW_VAL = {i + 1: (P16[i], P84[i]) for i in range(10)}

# (P0.13, P2.28, P97.72, P99.87) = (-3 sigma, -2 sigma, +2 sigma, +3 sigma)
# NEW_VAL = {
#     1:  (-2.6633, -2.2608,  1.6109, 3.6184),
#     2:  (-1.4626, -0.6727,  2.7320, 3.8584),
#     3:  (-1.8836, -0.9411,  1.0715, 1.5969),
#     4:  (-2.4405, -2.0222,  1.4067, 2.9262),
#     5:  (-5.0029, -2.3848,  0.9375, 1.3031),
#     6:  (-3.0621, -1.9341,  0.5589, 0.9225),
#     7:  (-1.4066, -1.1242,  1.2750, 2.5776),
#     8:  (-3.2852, -1.5618,  4.0277, 6.7008),
#     9:  (-3.6848, -1.9102,  1.7544, 2.7036),
#     10: (-2.4197, -1.2966,  1.3554, 2.1585),
# }
# COLORS = ['tab:purple', 'tab:blue', 'tab:green', 'tab:orange']
# LABELS = ['P0.13', 'P2.28', 'P97.72', 'P99.87']



# def vary_latent(latent, new_value):
#     variants = []
#     for i in range(10):
#         latent_update = latent.copy()
#         latent_update[i] = new_value
#         variants.append(latent_update)
#     return variants

# def set_latents(latent, changes):
#     latent_update = latent.copy()
#     for index, value in changes.items():
#         latent_update[index - 1] = value
#     return latent_update

def load_decoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sq = SpenderQ("qso.dr1.hiz")
    model = sq.models[-1][0]
    model.to(device)
    model.eval()
    return model, sq.wave_recon(), device


# take the function in the model
def decode(model, device, latent):
    s = torch.from_numpy(latent)[None, :].to(device)
    with torch.no_grad():
        recon = model.decode(s)
    return recon.cpu().numpy().squeeze()


def decode_sweep(model, device, base, index, values): # handles many latent at once
    batch = np.tile(base, (len(values), 1)).astype(np.float32)
    batch[:, index - 1] = values
    s = torch.from_numpy(batch).to(device)
    with torch.no_grad():
        recon = model.decode(s)
    return recon.cpu().numpy()

def decode_batch(model, device, Z):
    s = torch.from_numpy(np.asarray(Z, dtype=np.float32)).to(device)
    with torch.no_grad():
        return model.decode(s).cpu().numpy()

def set_latent(latent, index, value):
    latent_update = latent.copy()
    latent_update[index - 1] = value
    return latent_update

def plot_latent_sweep(wave_rest, curves, values, index, out_dir):
    norm = Normalize(vmin=values.min(), vmax=values.max())
    cmap = cm.viridis

    fig, ax = plt.subplots(figsize=(10, 4))
    for curve, v in zip(curves, values):
        ax.plot(wave_rest, curve, color=cmap(norm(v)), lw=1)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=f"Latent{index} value (16th to 84th percentile)")

    ax.set_xlabel("Rest wavelength [A]")
    ax.set_ylabel("Normalized flux")
    ax.set_title(f"Varying Latent{index}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"latent{index}_sweep.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path

def vary_all_latents(model, device, wave_rest, base, p16, p84, n_sweep, out_root):
    for index in range(1, len(base) + 1):
        values = np.linspace(p16[index - 1], p84[index - 1], n_sweep)
        curves = decode_sweep(model, device, base, index, values)
        path = plot_latent_sweep(wave_rest, curves, values, index,
                                os.path.join(out_root, str(index)))
        print("saved", path)

def line_sweep(X, latents, n=20, lo=0.1, hi=99.9, k=30):
    """Points along the main axis of the cloud for the given latents (1-based).
    Pair values lie exactly on the line; the other latents are the median of the
    k targets closest along the line, so they follow the data."""
    idx = [j - 1 for j in latents]
    med = np.median(X, 0)
    sig = (np.percentile(X, 84, 0) - np.percentile(X, 16, 0)) / 2
    Z = (X[:, idx] - med[idx]) / sig[idx]
    c = Z.mean(0)
    w, v = np.linalg.eigh(np.cov(Z.T))
    axis = v[:, -1]
    if axis[0] < 0:
        axis = -axis
    proj = (Z - c) @ axis
    t = np.linspace(*np.percentile(proj, [lo, hi]), n)
    line_Z = c + t[:, None] * axis
    vecs = np.empty((n, X.shape[1]))
    local_n = np.empty(n, int)
    for i, ti in enumerate(t):
        nn = np.argsort(np.abs(proj - ti))[:k]
        vecs[i] = np.median(X[nn], 0)
        local_n[i] = (np.linalg.norm(Z - line_Z[i], axis=1) < 0.5).sum()
    vecs[:, idx] = line_Z * sig[idx] + med[idx]
    return vecs.astype(np.float32), local_n

def vary_each_latent(model, device, wave_rest, default, offsets, out_dir, tag,
                     default_label="default epoch", other=None, other_label="other epoch",
                     scale=None):
    """For each latent in turn, add each offset to the default vector (others fixed),
    decode, and plot with the default epoch and (optionally) another epoch overlaid.
    scale: optional per-latent widths; if given, offsets are in units of scale[k]."""
    os.makedirs(out_dir, exist_ok=True)
    default = np.asarray(default, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    base_curve = decode_batch(model, device, default[None, :])[0]
    other_curve = None
    if other is not None:
        other_curve = decode_batch(model, device, np.asarray(other, dtype=np.float32)[None, :])[0]

    colors = sns.color_palette("coolwarm", len(offsets))
    paths = []
    for k in range(len(default)):
        step = offsets if scale is None else offsets * scale[k]
        Z = np.tile(default, (len(offsets), 1))
        Z[:, k] += step
        curves = decode_batch(model, device, Z)

        fig, ax = plt.subplots(figsize=(10, 4))
        for i, curve in enumerate(curves):
            ax.plot(wave_rest, curve, color=colors[i], lw=0.8, alpha=0.7,
                    label=f"{offsets[i]:+.1f}  (L{k+1}={Z[i, k]:.2f})")
        ax.plot(wave_rest, base_curve, color="black", lw=1.3, label=default_label)
        if other_curve is not None:
            ax.plot(wave_rest, other_curve, color="tab:green", lw=1.3, ls="--", label=other_label)
        ax.set_xlabel("Rest wavelength [A]")
        ax.set_ylabel("Flux")
        ax.set_title(f"{tag}: Latent{k+1} varied")
        ax.legend(ncol=2, fontsize=6)
        path = os.path.join(out_dir, f"{tag}_latent{k+1}_offsets.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths

if __name__ == "__main__":
    model, wave_rest, device = load_decoder()
    base = decode(model, device, latent_default)

    # # 1. specified values, one latent at a time
    # fig, ax = plt.subplots(figsize=(10, 4))
    # ax.plot(wave_rest, base, color='gray', label="median latents")
    # out_path = os.path.join(out_root, f"default.png")
    # ax.legend()
    # fig.savefig(out_path, dpi=150, bbox_inches='tight')
    # plt.close(fig)

    # for n_value in args.n_values:
    #     NEW_VAL = {k: np.linspace(lo, hi, n_value) for k, (lo, hi) in RANGES.items()}
    #     COLORS = sns.color_palette("RdBu_r", n_value)

    #     for index, values in NEW_VAL.items():
    #         out_dir = os.path.join(out_root, str(index))
    #         os.makedirs(out_dir, exist_ok=True)
    #         out_path = os.path.join(out_dir, f"latent{index}_values_{n_value}points_min_to_max.png")

    #         fig, ax = plt.subplots(figsize=(10, 4))

    #         curves = decode_sweep(model, device, latent_default, index, values)  
    #         for i, (curve, value) in enumerate(zip(curves, values)):
    #             ax.plot(wave_rest, curve, color=COLORS[i], alpha=0.6, lw=0.8, label=f"{value:.2f}")
    #         ymin, ymax = curves.min(), curves.max()
    #         pad = 0.05 * (ymax - ymin)
    #         ax.set_ylim(ymin - pad, ymax + pad)
    #         ax.set_xlabel("Rest wavelength [A]")
    #         ax.set_ylabel("Flux")
    #         ax.set_title(f"Latent{index}")
    #         ax.legend(ncol=2, fontsize=6)
    #         fig.savefig(out_path, dpi=300, bbox_inches='tight')
    #         plt.close(fig)
    #         print("saved", out_path)

    # 2. finer variation with colorbar, all 10 latents
   #vary_all_latents(model, device, wave_rest, latent_default, P16, P84, N_SWEEP, out_root)

    # #  3. pairs along the data line
    # X = latent_all.groupby("TARGETID")[cols].median().values
    # n_pair = 10
    # colors = sns.color_palette("coolwarm", n_pair)
    # pair_dir = os.path.join(out_root, "pairs")
    # os.makedirs(pair_dir, exist_ok=True)

    # for a, b in [(1, 4)]:                      # add more pairs here
    #     vecs, local_n = line_sweep(X, (a, b), n=n_pair, lo=5, hi=95)
    #     curves = decode_batch(model, device, vecs)

    #     fig, ax = plt.subplots(figsize=(10, 4))
    #     for i, curve in enumerate(curves):
    #         ax.plot(wave_rest, curve, color=colors[i], lw=0.9, alpha=0.8,
    #                 label=f"L{a}={vecs[i, a-1]:.2f}, L{b}={vecs[i, b-1]:.2f} (n≈{local_n[i]})")
    #     ax.set_xlabel("Rest wavelength [A]")
    #     ax.set_ylabel("Flux")
    #     ax.set_title(f"Latent{a} + Latent{b} ")
    #     ax.legend(ncol=2, fontsize=6)
    #     out_path = os.path.join(pair_dir, f"pair_L{a}_L{b}_{n_pair}points.png")
    #     fig.savefig(out_path, dpi=300, bbox_inches="tight")
    #     plt.close(fig)
    #     print("saved", out_path)

    clq_dir = os.path.join(out_root, "clq")
    vary_each_latent(model, device, wave_rest,
                    latent_default__39633100518459964, OFFSETS, clq_dir,
                    "CLQ39633100518459964",
                    default_label="20220529 (default)",
                    other=latent_other__39633100518459964, other_label="20210519")

    two = decode_batch(model, device, np.stack([latent_other__39633100518459964,
                                            latent_default__39633100518459964]))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wave_rest, two[0], color="tab:green", label="20210519")
    ax.plot(wave_rest, two[1], color="black", label="20220529")
    ax.legend(); ax.set_xlabel("Rest wavelength [A]"); ax.set_ylabel("Flux")
    fig.savefig(os.path.join(clq_dir, "CLQ39633100518459964_two_epochs.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)