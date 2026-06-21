import os
import sys

# Makes imports work both as `python model/train.py` and `python -m model.train`.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import argparse
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from data.synthetic_generator import generate_synthetic_mirna_dataset
from data.loader import load_expression_csv, preprocess_train, augment_data
from data.geo_soft_loader import load_multiple_geo_soft
from model.transformer import TabularMiRNATransformer


def train(
    csv_path=None,
    soft_paths=None,
    label_col="label",
    epochs=8,
    batch_size=32,
    lr=1e-3,
    synthetic=True,
    augment=False,
    out_dir="artifacts",
    max_features=512,
):
    os.makedirs(out_dir, exist_ok=True)

    if soft_paths:
        Xdf, y, _meta_samples = load_multiple_geo_soft(soft_paths)
        print("Loaded GEO SOFT samples:", Xdf.shape, "labels:", y.value_counts().to_dict())
    elif csv_path:
        Xdf, y = load_expression_csv(csv_path, label_col)
        if y is None:
            raise ValueError("CSV must contain labels for training.")
    elif synthetic:
        Xdf, y = generate_synthetic_mirna_dataset(n_samples=800, n_features=200)
    else:
        raise ValueError("Provide csv_path, soft_paths, or synthetic=True")

    X, y_enc, meta = preprocess_train(Xdf, y, max_features=max_features)
    print(f"Training matrix after preprocessing: {X.shape}")

    if augment:
        X, y_enc = augment_data(X, y_enc)

    stratify = y_enc if len(set(y_enc)) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y_enc, test_size=0.2, stratify=stratify, random_state=42
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.long)),
        batch_size=batch_size,
        shuffle=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabularMiRNATransformer(X.shape[1], len(meta["label_encoder"].classes_)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        print(f"epoch={epoch + 1} loss={total / max(len(Xtr), 1):.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xte, dtype=torch.float32).to(device)).cpu()
        pred = logits.argmax(1).numpy()

    print("Accuracy:", accuracy_score(yte, pred))
    print(classification_report(yte, pred, target_names=meta["label_encoder"].classes_, zero_division=0))

    torch.save(model.state_dict(), os.path.join(out_dir, "mirna_transformer.pt"))
    joblib.dump(meta, os.path.join(out_dir, "preprocess_meta.joblib"))
    print("Saved artifacts to:", os.path.abspath(out_dir))
    return model, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", default=None)
    p.add_argument("--soft_paths", nargs="+", default=None)
    p.add_argument("--label_col", default="label")
    p.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", "8")))
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--out_dir", default="artifacts")
    p.add_argument("--max_features", type=int, default=int(os.environ.get("MAX_FEATURES", "512")))
    p.add_argument("--synthetic", action="store_true", default=True)
    args = p.parse_args()
    train(**vars(args))
