import numpy as np
import pandas as pd
from typing import Tuple, List

CANCER_CLASSES = ["normal", "lung_cancer", "breast_cancer", "colon_cancer"]
KNOWN_MIRNAS = ["miR-21", "miR-155", "miR-34a", "miR-210", "miR-200c", "miR-145", "miR-16", "miR-31", "miR-92a", "miR-10b"]


def make_mirna_ids(n_features: int) -> List[str]:
    ids = KNOWN_MIRNAS.copy()
    i = 1
    while len(ids) < n_features:
        ids.append(f"miR-{1000+i}")
        i += 1
    return ids[:n_features]


def generate_synthetic_mirna_dataset(
    n_samples: int = 800,
    n_features: int = 200,
    random_state: int = 42,
    class_balance: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Generate realistic-ish blood microRNA expression data.

    Values simulate normalized log2 expression with cancer-specific biomarker shifts,
    batch noise, donor variability, and random biological noise.
    """
    rng = np.random.default_rng(random_state)
    feature_names = make_mirna_ids(n_features)
    n_classes = len(CANCER_CLASSES)

    if class_balance:
        y = np.repeat(np.arange(n_classes), n_samples // n_classes)
        if len(y) < n_samples:
            y = np.concatenate([y, rng.choice(n_classes, n_samples - len(y))])
        rng.shuffle(y)
    else:
        y = rng.choice(n_classes, size=n_samples, p=[0.45, 0.2, 0.2, 0.15])

    base_means = rng.normal(loc=8.0, scale=1.2, size=n_features)
    base_scales = rng.uniform(0.4, 1.1, size=n_features)
    X = rng.normal(base_means, base_scales, size=(n_samples, n_features))

    # Global donor and batch effects
    X += rng.normal(0, 0.35, size=(n_samples, 1))
    batch_effect = rng.normal(0, 0.2, size=(4, n_features))
    batches = rng.integers(0, 4, size=n_samples)
    X += batch_effect[batches]

    # Cancer-specific biomarker signatures
    signatures = {
        1: {"miR-21": 1.8, "miR-155": 1.2, "miR-210": 1.5, "miR-34a": -0.9},
        2: {"miR-21": 1.4, "miR-10b": 1.5, "miR-200c": -1.1, "miR-145": -0.8},
        3: {"miR-92a": 1.6, "miR-31": 1.4, "miR-145": -1.2, "miR-16": -0.6},
    }
    idx = {f: i for i, f in enumerate(feature_names)}
    for cls, effects in signatures.items():
        rows = np.where(y == cls)[0]
        for mirna, shift in effects.items():
            if mirna in idx:
                X[rows, idx[mirna]] += shift + rng.normal(0, 0.25, size=len(rows))

    # Sparse hidden signals across additional features
    for cls in range(1, n_classes):
        rows = np.where(y == cls)[0]
        signal_cols = rng.choice(np.arange(10, n_features), size=min(15, max(1, n_features - 10)), replace=False)
        X[np.ix_(rows, signal_cols)] += rng.normal(0.6, 0.25, size=(len(rows), len(signal_cols)))

    X = np.clip(X, 0, None)
    df = pd.DataFrame(X, columns=feature_names)
    labels = pd.Series([CANCER_CLASSES[i] for i in y], name="label")
    return df, labels


if __name__ == "__main__":
    X, y = generate_synthetic_mirna_dataset()
    out = X.copy()
    out["label"] = y
    out.to_csv("synthetic_mirna_dataset.csv", index=False)
    print(out.shape)
