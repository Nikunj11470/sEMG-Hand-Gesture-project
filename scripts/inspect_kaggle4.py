import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1])
ch = [f"channel{i}" for i in range(1, 9)]
d = df[(df["label"] == 1)]
x = d[ch].to_numpy(); t = d["time"].to_numpy()
new = np.r_[True, np.any(x[1:] != x[:-1], axis=1)]
gaps = np.diff(t[new]); gaps = gaps[gaps > 0]
print("ms between changed rows: median", np.median(gaps), "| quantiles", np.quantile(gaps, [.1, .25, .75, .9]).tolist())
print("new samples per second:", round(new.sum() / (t.max() / 1000 * 2), 1))
