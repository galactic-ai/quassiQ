import os
import sys
sys.path.append("/work/11161/kanyuni/ls6/quassiQ_project/spenderq/src")

import numpy as np
import torch
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from spenderq.spenderq import SpenderQ

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
n_value = 10 
NEW_VAL = {
    1:  np.linspace(-3.1142, 5.1101, n_value),
    2:  np.linspace(-1.9557, 4.2888, n_value),
    3:  np.linspace(-3.3126, 2.0338, n_value),
    4:  np.linspace(-2.8105,5.2354, n_value),
    5:  np.linspace(-7.5185, 1.8542, n_value),
    6:  np.linspace(-4.2829, 1.3293, n_value),
    7:  np.linspace(-1.6595, 3.7149, n_value),
    8:  np.linspace(-4.2252, 7.6572, n_value),
    9:  np.linspace(-5.5418, 3.8828, n_value),
    10: np.linspace(-3.3234, 2.5936, n_value),
}

COLORS = sns.color_palette("tab20", n_value)

# N_SWEEP = 21 
# from corner plot (latent_corner.py)
P16 = np.array([-1.8213,  0.0229, -0.1790, -1.4598, -0.6372,
                -0.9653, -0.8453, -0.0861, -0.3044, -0.3705], dtype=np.float32)
MEDIAN = np.array([-1.2561,  0.6103,  0.2944, -0.8168,  0.1343,
                   -0.3024, -0.5056,  0.8830,  0.5331,  0.2229], dtype=np.float32)
latent_default = MEDIAN.copy()  
P84 = np.array([-0.2580,  1.3265,  0.6263,  0.0710,  0.5901,
                 0.1626,  0.0745,  2.0186,  0.9916,  0.7154], dtype=np.float32)

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


if __name__ == "__main__":
    model, wave_rest, device = load_decoder()
    base = decode(model, device, latent_default)

    # 1. specified values, one latent at a time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(wave_rest, base, color='gray', label="median latents")
    out_path = os.path.join(out_root, f"default_model.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    ax.legend()
    plt.close(fig)
    for index, values in NEW_VAL.items():
        out_dir = os.path.join(out_root, str(index))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"latent{index}_values_{n_value}points_min_to_max.png")

        fig, ax = plt.subplots(figsize=(10, 4))

        curves = decode_sweep(model, device, latent_default, index, values)  
        for i, (curve, value) in enumerate(zip(curves, values)):
            ax.plot(wave_rest, curve, color=COLORS[i], alpha=0.6, lw=0.8, label=f"{value:.2f}")
        ax.set_xlabel("Rest wavelength [A]")
        ax.set_ylabel("Flux")
        ax.set_title(f"Latent{index}")
        ax.legend(ncol=2, fontsize=6)
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("saved", out_path)

    # 2. finer variation with colorbar, all 10 latents
   #vary_all_latents(model, device, wave_rest, latent_default, P16, P84, N_SWEEP, out_root)