"""Frequency plot for Kaggle: raw 1 ms rows vs the 200 Hz grid vs filtered.

Run from the project root:
    python -m scripts.plot_kaggle_spectrum
Saves docs/figures/kaggle_spectrum.png and kaggle_filter_example.png.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from src.data import signal as sig
from src.data.kaggle import CHANNELS, FS, load_csv

OUT = "docs/figures"
CH = 0  # channel1


def main():
    df = load_csv("data/raw/kaggle/EMG-data.csv")
    g = df[(df["label"] == 1) & (df["rec"] == 0)]
    t, raw = g["time"].to_numpy(), g[CHANNELS].to_numpy(np.float64)
    _, uni = sig.to_uniform(t, raw, FS)
    filt = sig.bandpass(uni, FS, 20, 90)

    repeats = np.all(raw[1:] == raw[:-1], axis=1).mean()
    os.makedirs(OUT, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for x, fs, name in [(raw, 1000, "raw rows, read as 1000 Hz"),
                        (uni, FS, "on a 200 Hz grid"),
                        (filt, FS, "200 Hz + 20-90 Hz bandpass")]:
        f, p = welch(x[:, CH] - x[:, CH].mean(), fs=fs, nperseg=512)
        ax.semilogy(f, p, label=name, lw=1.2)
    ax.axvline(50, color="gray", ls=":", lw=1)
    ax.text(51, ax.get_ylim()[1] * 0.3, "50 Hz", color="gray")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power")
    ax.set_title(f"Kaggle, person 1, channel 1: {repeats:.0%} of raw rows repeat the previous row")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/kaggle_spectrum.png", dpi=110)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    seg = slice(0, 2 * FS)
    for ax, x, name in [(axes[0], uni, "200 Hz grid, unfiltered"), (axes[1], filt, "20-90 Hz bandpass")]:
        ax.plot(np.arange(len(x[seg])) / FS, x[seg, CH], lw=0.8)
        ax.set_title(name)
    axes[1].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(f"{OUT}/kaggle_filter_example.png", dpi=110)
    print(f"saved {OUT}/kaggle_spectrum.png and {OUT}/kaggle_filter_example.png")


if __name__ == "__main__":
    main()
