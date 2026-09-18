import os
import pandas as pd
import matplotlib.pyplot as plt

files = {
    r"Ly$\alpha$": "/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/lya/batch/lya_first_1000_results.csv",
    "Mg II": "/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/mgii/batch/mgii_first_1000_results.csv",
    "C IV": "/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/civ/batch/civ_first_1000_results.csv",
}

output_dir = "/work/11161/kanyuni/ls6/quassiQ_project/pipeline_output/overlap"
os.makedirs(output_dir, exist_ok=True)

for label, path in files.items():
    df = pd.read_csv(path)
    plt.hist(
        df["peak_n_sigma"].dropna(),
        bins=30,
        histtype="step",
        linewidth=2,
        label=label,
    )

plt.axvline(3, color="black", linestyle="--", label=r"$N_\sigma = 3$")
plt.xlabel(r"Peak $N_\sigma$")
plt.ylabel("Number of targets")
plt.legend()
plt.tight_layout()

output_path = os.path.join(output_dir, "emission_line_nsigma_histogram.png")
plt.savefig(output_path, dpi=200)
plt.close()

print(f"Saved: {output_path}")