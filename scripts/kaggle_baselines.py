import h5py
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import f1_score

with h5py.File("data/processed/kaggle/data.h5") as f:
    X, y, subj = f["X"][:], f["y"][:], f["subject"][:]

def features(X):
    d = np.diff(X, axis=1)
    return np.concatenate([np.abs(X).mean(1), np.sqrt((X**2).mean(1)), np.abs(d).sum(1),
                           (X[:, :-1] * X[:, 1:] < 0).sum(1), (d[:, :-1] * d[:, 1:] < 0).sum(1),
                           X.max(1) - X.min(1)], axis=1)
F = features(X)

models = {
    "KNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "SVM": lambda: make_pipeline(StandardScaler(), SVC()),
    "RF":  lambda: RandomForestClassifier(200, n_jobs=-1, random_state=0),
}
idx = np.arange(len(y))
rows = []
for name, make in models.items():
    for split in ["random", "new people"]:
        scores = []
        for seed in range(5):
            if split == "random":
                tr, te = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
            else:
                tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(F, y, subj))
            pred = make().fit(F[tr], y[tr]).predict(F[te])
            scores.append(f1_score(y[te], pred, average="macro"))
        rows.append((name, split, np.mean(scores), np.std(scores)))
        print(f"{name:4s} {split:11s} macro-F1 {np.mean(scores):.1%} ± {np.std(scores):.1%}")
