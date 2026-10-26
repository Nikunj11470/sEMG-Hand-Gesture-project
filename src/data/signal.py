"""Shared sEMG signal processing used by every dataset loader.

All functions take signals shaped (T, C): T time steps, C channels.
Filter the whole continuous recording first, then cut windows.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly


def bandpass(x, fs, low=20.0, high=None, order=4):
    """Butterworth bandpass, zero phase. `high` defaults to 0.45 * fs (under Nyquist)."""
    high = high or 0.45 * fs
    if not 0 < low < high < fs / 2:
        raise ValueError(f"need 0 < low < high < fs/2, got low={low}, high={high}, fs={fs}")
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def highpass(x, fs, cutoff=20.0, order=4):
    """Butterworth highpass, zero phase. Removes drift and DC offset."""
    sos = butter(order, cutoff, btype="high", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def notch(x, fs, f0=50.0, q=30.0):
    """Remove power-line hum at f0 (50 or 60 Hz). Skipped if f0 is above Nyquist."""
    if f0 >= fs / 2:
        return x
    b, a = iirnotch(f0, q, fs=fs)
    return filtfilt(b, a, x, axis=0)


def resample(x, fs_in, fs_out):
    """Resample an evenly sampled signal from fs_in to fs_out (integer rates)."""
    if fs_in == fs_out:
        return x
    g = np.gcd(int(fs_in), int(fs_out))
    return resample_poly(x, int(fs_out) // g, int(fs_in) // g, axis=0)


def to_uniform(t_ms, x, fs_out, labels=None):
    """Put an unevenly timed signal on an even grid of fs_out Hz.

    Takes the last sample at or before each grid time (sample-and-hold), which is
    how the Kaggle Myo data is stored. Returns (t_grid_ms, x_grid[, labels_grid]).
    """
    t_ms = np.asarray(t_ms)
    t_grid = np.arange(t_ms[0], t_ms[-1], 1000.0 / fs_out)
    idx = np.searchsorted(t_ms, t_grid, side="right") - 1
    if labels is None:
        return t_grid, x[idx]
    return t_grid, x[idx], np.asarray(labels)[idx]


def preprocess(x, fs, low=20.0, high=None, notch_hz=50.0, fs_out=None):
    """The one call loaders use: notch -> bandpass -> resample.

    Pass low=None to skip the bandpass, notch_hz=None to skip the notch.
    """
    if notch_hz:
        x = notch(x, fs, notch_hz)
    if low:
        x = bandpass(x, fs, low, high)
    if fs_out and fs_out != fs:
        x = resample(x, fs, fs_out)
    return x


def make_windows(x, labels, fs, window_ms=200, step_ms=100, keep=None):
    """Cut (T, C) into windows of window_ms every step_ms.

    A window is kept only if all its labels are the same, and (if `keep` is given)
    that label is in `keep`. Returns windows (N, W, C), labels (N,), starts (N,).
    """
    win = int(round(window_ms * fs / 1000))
    step = int(round(step_ms * fs / 1000))
    labels = np.asarray(labels)
    out_x, out_y, starts = [], [], []
    for s in range(0, len(x) - win + 1, step):
        lab = labels[s:s + win]
        if (lab != lab[0]).any():
            continue
        if keep is not None and lab[0] not in keep:
            continue
        out_x.append(x[s:s + win])
        out_y.append(lab[0])
        starts.append(s)
    if not out_x:
        return np.empty((0, win, x.shape[1]), np.float32), np.empty(0, int), np.empty(0, int)
    return np.stack(out_x).astype(np.float32), np.array(out_y), np.array(starts)
