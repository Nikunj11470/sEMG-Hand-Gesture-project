import sys
import numpy as np
import pandas as pd
from scipy.signal import welch

df = pd.read_csv(sys.argv[1])
ch = [f"channel{i}" for i in range(1, 9)]
x = df[df["label"] == 1][ch].to_numpy()
same = np.all(x[1:] == x[:-1], axis=1).mean()
print(f"rows identical to the previous row: {same:.1%}")
f, p = welch(x[:, 0] - x[:, 0].mean(), fs=1000, nperseg=1024)
tot = p.sum()
for lo, hi in [(0, 20), (20, 100), (100, 200), (200, 500)]:
    print(f"power {lo}-{hi} Hz: {p[(f >= lo) & (f < hi)].sum() / tot:.1%}")
