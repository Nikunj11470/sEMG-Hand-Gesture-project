# sEMG-Hand-Gesture-project
 sEMG sensors on the forearm record the electrical activity of muscles.

We classify hand gestures from short windows of forearm sEMG and test which models stay accurate on **new people** and **new days**: hand-made features with KNN / SVM / Random Forest, 1D CNN, GRU, Transformer and Mamba. See the project proposal PDF for the full plan.

## Setup

```bash
pip install -r requirements.txt
```

Datasets are not stored in this repo. Download them into `data/raw/<dataset>/`:

| Dataset | Link | Put it in |
| --- | --- | --- |
| Kaggle EMG gesture (UCI Myo, 36 people, 8 channels) | [kaggle](https://www.kaggle.com/datasets/sojanprajapati/emg-signal-for-gesture-recognition) | `data/raw/kaggle/EMG-data.csv` |
| Mendeley multi-channel sEMG (40 people, 4 channels) | [mendeley](https://data.mendeley.com/datasets/ckwc76xr2z/2) | `data/raw/mendeley/` |
| GRABMyo (43 people, 3 days, ~9.4 GB) | [physionet](https://physionet.org/content/grabmyo/1.1.0/) | `data/raw/grabmyo/` |

## Build the processed data

Run from the repo root. Each loader writes `data/processed/<dataset>/data.h5` with:
`X` (windows, time, channels) float32, `y` gesture 0..K-1, `subject`, `session`, `trial`.

```bash
python -m src.data.kaggle
```

## Layout

```
src/data/signal.py    shared filters, resampling, windowing (use this in every loader)
src/data/kaggle.py    Kaggle CSV -> data.h5
src/data/dataset.py   PyTorch Dataset: data.h5 + split file, train-only normalisation
scripts/              plots and the classical baseline
docs/figures/         plots
```

## Results so far (Kaggle, macro-F1, 5 seeds)

| Model | Random split | New people |
| --- | --- | --- |
| KNN | 89.7% | 76.2% ± 4.5 |
| SVM | 90.9% | 81.7% ± 5.4 |
| Random Forest | 93.7% | 83.8% ± 4.4 |

Kaggle notes: the CSV has a row every 1 ms but ~90% of rows repeat, so the real rate is ~100-200 Hz; the loader resamples to 200 Hz (200 ms window = 40 samples). There is no 50 Hz hum and bandpass filtering does not change the result, so the default is no filter.
