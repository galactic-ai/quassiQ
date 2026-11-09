import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner

latent_all_path = "/work/10579/prisha/ls6/desi_project/latent_info.csv"
latent_all = pd.read_csv(latent_all_path, dtype={"TARGETID": "int64"})

cols = [f"Latent{i}" for i in range(1, 11)]
L = latent_all[cols].dropna().values.astype(np.float32)
mean = np.mean(L, axis=0)
p16, median, p84 = np.percentile(L, [16, 50, 84], axis=0)
p2_5, p97_5 = np.percentile(L, [2.28, 97.72], axis=0)     # ±2σ
p0_13, p99_87 = np.percentile(L, [0.13, 99.87], axis=0)   # ±3σ
lmax = np.max(L, axis=0)
lmin = np.min(L, axis=0)

fig = corner.corner(
    L,
    labels=cols,
    color = "tab:blue",
    fill_contours=True,
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 8},
    label_kwargs={"fontsize": 8},
    plot_datapoints=True,
    plot_density=True,
    bins=30,
)
for name, a, b, c, d in zip(cols, p2_5, p97_5, p0_13, p99_87):
    print(f"{name:9s} 2\u03c3: {a:8.4f} to {b:8.4f}   3\u03c3: {c:8.4f} to {d:8.4f}")

corner.overplot_lines(fig, median, color="crimson", linestyle="--", linewidth=1)
corner.overplot_points(fig, median[None, :], marker="s", color="crimson")

corner.overplot_lines(fig, mean, color="tab:orange", linestyle="--", linewidth=1)
corner.overplot_points(fig, mean[None, :], marker="s", color="tab:orange")
handles = [
    Line2D([0], [0], color="crimson", ls="--", lw=1, marker="s", ms=6, label="Median"),
    Line2D([0], [0], color="tab:orange", ls="--", lw=1, marker="s", ms=6, label="Mean"),
]
fig.legend(handles=handles, loc="upper right",
           bbox_to_anchor=(0.95, 0.95), fontsize=20, frameon=True)

fig.savefig("/work/11161/kanyuni/ls6/quassiQ_project/latent/distribution/latent_corner.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved latent_corner.png", L.shape)