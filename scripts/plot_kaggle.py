"""Plot 2 s of all 8 channels per gesture for one person, plus class balance."""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

NAMES = {1: "rest", 2: "fist", 3: "wrist flexion", 4: "wrist extension",
         5: "radial dev.", 6: "ulnar dev."}
df = pd.read_csv(sys.argv[1])
ch = [f"channel{i}" for i in range(1, 9)]
p = df[df["label"] == (int(sys.argv[2]) if len(sys.argv) > 2 else 1)].copy()
# one id per continuous run of the same class, so a plot never spans two repetitions
p["run"] = (p["class"] != p["class"].shift()).cumsum()

fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
for ax, (c, name) in zip(axes.flat, NAMES.items()):
    seg = p[p["run"] == p.loc[p["class"] == c, "run"].iloc[0]]
    for i, col in enumerate(ch):
        ax.plot((seg["time"] - seg["time"].iloc[0]) / 1000, seg[col] + i * 2e-4, lw=0.6)
    ax.set_title(f"class {c}: {name}")
    ax.set_yticks([i * 2e-4 for i in range(8)], [f"ch{i+1}" for i in range(8)])
    ax.set_xlabel("s")
fig.suptitle("Kaggle EMG, person 1: first repetition of each gesture (channels offset)")
fig.tight_layout(); fig.savefig("kaggle_gestures_person1.png", dpi=110)

fig, ax = plt.subplots(figsize=(7, 4))
df["class"].value_counts().sort_index().plot.bar(ax=ax)
ax.set_xlabel("class"); ax.set_ylabel("rows"); ax.set_title("Rows per class (0 = between gestures)")
fig.tight_layout(); fig.savefig("kaggle_class_balance.png", dpi=110)
