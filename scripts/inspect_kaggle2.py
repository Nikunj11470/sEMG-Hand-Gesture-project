import sys
import pandas as pd

df = pd.read_csv(sys.argv[1])
dt = df.groupby("label")["time"].diff()
print("time step (ms): median", dt.median(), "| quantiles", dt.quantile([.05, .25, .75, .95]).tolist())
print("backward jumps (new recording starts):", int((dt < 0).sum()))
df["rec"] = (dt < 0).groupby(df["label"]).cumsum()
recs = df.groupby(["label", "rec"]).size()
print("recordings per person:", recs.groupby(level=0).size().value_counts().to_dict())
print("rows per recording: min", recs.min(), "max", recs.max())
print("duration per recording (s): ~", (df.groupby(["label","rec"])["time"].max() / 1000).describe().round(1).to_dict())
print("\nclass counts:\n", df["class"].value_counts().sort_index().to_string())
c7 = df[df["class"] == 7]["label"].unique()
print("\npeople with class 7:", len(c7), sorted(c7))
