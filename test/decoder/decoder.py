import os
import sys
sys.path.append("/work/11161/kanyuni/ls6/quassiQ_project/spenderq/src")

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from spenderq.spenderq import SpenderQ

latent_default = np.array([0.1] * 10, dtype=np.float32)
NEW_VAL = {3: 0.5, 7: -0.2}
VALUES = np.linspace(-2, 2, 21) 
out_root = "/work/11161/kanyuni/ls6/quassiQ_project/decoder"

def vary_latent(latent, new_value):
    variants = []
    for i in range(10):
        latent_update = latent.copy()
        latent_update[i] = new_value
        variants.append(latent_update)
    return variants

def set_latents(latent, changes):
    latent_update = latent.copy()
    for index, value in changes.items():
        latent_update[index - 1] = value
    return latent_update

def load_decoder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sq = SpenderQ("qso.dr1.hiz")
    model = sq.models[-1][0]
    model.to(device)
    model.eval()
    return model, sq.wave_recon(), device


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
    fig.colorbar(sm, ax=ax, label=f"Latent{index} value")

    ax.set_xlabel("Rest wavelength [A]")
    ax.set_ylabel("Normalized flux")
    ax.set_title(f"Varying Latent{index}")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"latent{index}_sweep.png")
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


def vary_all_latents(model, device, wave_rest, base, values, out_root):
    """Wrapper: one colorbar sweep plot for each of the 10 latents."""
    for index in range(1, len(base) + 1):
        curves = decode_sweep(model, device, base, index, values)
        path = plot_latent_sweep(wave_rest, curves, values, index,
                                 os.path.join(out_root, str(index)))
        print("saved", path)


if __name__ == "__main__":
    model, wave_rest, device = load_decoder()
    base = decode(model, device, latent_default)

    # 1. specified values, one latent at a time
    for index, value in NEW_VAL.items():
        changed = decode(model, device, set_latent(latent_default, index, value))

        out_dir = os.path.join(out_root, str(index))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"latent{index}_{value}.png")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(wave_rest, base, color='gray', label="default")
        ax.plot(wave_rest, changed, color='tab:red', label=f"Latent{index} = {value}")
        ax.set_xlabel("Rest wavelength [A]")
        ax.set_ylabel("Normalized flux")
        ax.legend()
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("saved", out_path)

    # 2. finer variation with colorbar, all 10 latents
    vary_all_latents(model, device, wave_rest, latent_default, VALUES, out_root)