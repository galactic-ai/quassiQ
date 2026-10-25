import pandas as pd

csv = {
    "score1": "/work/11161/kanyuni/ls6/quassiQ_project/latent/revisit_newpipeline/score1_latent.csv",
    "score5": "/work/11161/kanyuni/ls6/quassiQ_project/latent/revisit_newpipeline/score5_latent.csv",
}
latent_cols = [f"Latent{i}" for i in range(1, 11)]

df1 = pd.read_csv(csv["score1"], dtype={"TARGETID": "int64"})
df5 = pd.read_csv(csv["score5"], dtype={"TARGETID": "int64"})
df1["is5"] = 0
df5["is5"] = 1
df = pd.concat([df1, df5], ignore_index=True)


def report(name, data, selected):
    """selected = True/False per row. Prints how many kept, purity, recall."""
    n_kept = selected.sum()
    purity = data["is5"][selected].mean()                 # fraction of kept that are score 5
    recall = data["is5"][selected].sum() / data["is5"].sum()  # fraction of all score 5 kept
    print(f"{name:55s} kept={n_kept:3d}  selected 5 / selected 5 + selected 1 ={purity:.2f}  selected 5 / all 5={recall:.2f}")


# 1 Cuts on single spectra
print(" per-spectrum cuts ")
print("baseline purity (fraction of spectra that are score 5):", round(df["is5"].mean(), 2))

report("Latent1 <= -1.8", df, df["Latent1"] <= -1.8)
report("Latent1 <= -1.2", df, df["Latent1"] <= -1.2)
report("Latent1 <= -1.0", df, df["Latent1"] <= -1.0)
report("Latent4 <= -1.38", df, df["Latent4"] <= -1.38)
report("Latent1 <= -1.8 or Latent4 <= -1.38", df,
       (df["Latent1"] <= -1.8) | (df["Latent4"] <= -1.38))

# 2 Cuts on whole targets (combine all spectra of a target)
grouped = df.groupby("TARGETID")

targets = pd.DataFrame({
    "is5": grouped["is5"].first(),                    # same for every spectrum of a target
    "mean_L1": grouped["Latent1"].mean(),             # target-averaged Latent1
    "total_var": grouped[latent_cols].var().sum(axis=1),  # sum of within-target variances
})

print()
print(" per-target cuts ")
print("baseline purity (fraction of targets that are score 5):", round(targets["is5"].mean(), 2))

report("mean Latent1 <= -1.5", targets, targets["mean_L1"] <= -1.5)
report("mean Latent1 <= -1 and total_var >= 0.6", targets,
       (targets["mean_L1"] <= -1) & (targets["total_var"] >= 0.6))
report("mean Latent1 <= -1 and total_var >= 0.8", targets,
       (targets["mean_L1"] <= -1) & (targets["total_var"] >= 0.8))