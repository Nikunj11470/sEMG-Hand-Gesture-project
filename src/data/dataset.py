"""PyTorch Dataset over any data.h5 + a split file.

Split file format (written by splits.py):
    {"train": [window indices], "val": [...], "test": [...]}

Normalisation (per-channel mean and std) is computed on the TRAIN windows only and
reused for val and test, so no test information leaks into training.

Usage:
    train_ds = EMGDataset("data/processed/kaggle/data.h5", "data/processed/kaggle/split_subject.json", "train")
    test_ds  = EMGDataset(..., "test", stats=train_ds.stats)
"""
import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def load_split(split_path):
    with open(split_path) as f:
        return {k: np.asarray(v, dtype=np.int64) for k, v in json.load(f).items()}


def channel_stats(X):
    """Per-channel mean and std over all windows and time steps of X (N, T, C)."""
    mean = X.mean(axis=(0, 1))
    std = X.std(axis=(0, 1)) + 1e-8
    return mean.astype(np.float32), std.astype(np.float32)


class EMGDataset(Dataset):
    """Returns (x, y): x float32 tensor (T, C), y int64 label."""

    def __init__(self, h5_path, split_path, part, stats=None):
        idx = np.sort(load_split(split_path)[part])
        with h5py.File(h5_path, "r") as f:
            self.X = f["X"][idx].astype(np.float32)
            self.y = f["y"][idx].astype(np.int64)
            self.subject = f["subject"][idx]
            self.n_classes = int(f["y"][:].max()) + 1
        if stats is None:
            if part != "train":
                raise ValueError("pass stats=train_ds.stats for val/test so normalisation uses training data only")
            stats = channel_stats(self.X)
        self.stats = stats
        mean, std = stats
        self.X = (self.X - mean) / std

    @property
    def n_channels(self):
        return self.X.shape[2]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])
