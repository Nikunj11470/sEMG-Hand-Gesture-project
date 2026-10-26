"""Kaggle EMG gesture dataset (UCI Myo armband, 36 people, 8 channels) -> data.h5.

The CSV has a row every ~1 ms, but ~90% of rows repeat the previous one: the real
rate is ~100-200 Hz. So each recording is put on an even 200 Hz grid first.

Run from the project root:
    python -m src.data.kaggle
    python -m src.data.kaggle --filter bandpass
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd

from src.data import signal as sig

CHANNELS = [f"channel{i}" for i in range(1, 9)]
# UCI labels; 0 = between gestures and 7 = extended palm (only 2 people) are dropped
LABEL_NAMES = ["rest", "fist", "wrist flexion", "wrist extension", "radial deviation", "ulnar deviation"]
KEEP = range(1, 7)
FS = 200


def load_csv(path):
    df = pd.read_csv(path)
    # time restarts at each new recording; each person has 2
    df["rec"] = (df.groupby("label")["time"].diff() < 0).groupby(df["label"]).cumsum()
    return df


def build(csv_path, out_path, filt="none", window_ms=200, step_ms=100):
    df = load_csv(csv_path)
    X, y, subject, trial = [], [], [], []
    for (person, rec), g in df.groupby(["label", "rec"]):
        _, x, lab = sig.to_uniform(g["time"].to_numpy(), g[CHANNELS].to_numpy(np.float64), FS, g["class"].to_numpy())
        if filt == "highpass":
            x = sig.highpass(x, FS, 20)
        elif filt == "bandpass":
            x = sig.bandpass(x, FS, 20, 90)
        w, wl, _ = sig.make_windows(x, lab, FS, window_ms, step_ms, keep=KEEP)
        X.append(w)
        y.append(wl - 1)  # labels 1..6 -> 0..5
        subject.append(np.full(len(w), person))
        trial.append(np.full(len(w), rec))
    X, y = np.concatenate(X), np.concatenate(y)
    subject, trial = np.concatenate(subject), np.concatenate(trial)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f["X"], f["y"], f["subject"], f["trial"] = X, y, subject, trial
        f["session"] = np.zeros_like(subject)  # one recording day only
        f.attrs.update(fs=FS, window_ms=window_ms, step_ms=step_ms, filter=filt,
                       label_names=LABEL_NAMES, dataset="kaggle")
    return X, y, subject, trial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/kaggle/EMG-data.csv")
    ap.add_argument("--out", default="data/processed/kaggle/data.h5")
    ap.add_argument("--filter", choices=["none", "highpass", "bandpass"], default="none")
    args = ap.parse_args()
    X, y, subject, _ = build(args.csv, args.out, args.filter)
    print(f"saved {args.out}: X {X.shape}, {len(np.unique(subject))} people, windows per class {np.bincount(y).tolist()}")


if __name__ == "__main__":
    main()
