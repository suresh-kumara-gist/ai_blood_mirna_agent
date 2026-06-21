"""GEO SOFT (.soft/.soft.gz) loader for microRNA expression datasets.

Parses family SOFT files where each ^SAMPLE block contains an ID_REF/VALUE table.
It returns a sample x feature matrix plus harmonized labels.
"""
from __future__ import annotations

import gzip
import os
import re
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _open_text(path: str):
    return gzip.open(path, "rt", errors="ignore") if path.endswith(".gz") else open(path, "rt", errors="ignore")


def infer_label_from_title(title: str, source: str = "") -> Optional[str]:
    text = f"{title} {source}".lower()
    # normals / controls first
    if any(k in text for k in ["non-cancer", "non cancer", "normal", "control", "healthy"]):
        return "normal"
    if "benign" in text or "borderline" in text:
        return None  # exclude ambiguous non-malignant disease by default
    if any(k in text for k in ["lung", "adenocarcinoma", "squamous", "small cell", "bronchioloalveolar"]):
        return "lung_cancer"
    if "breast" in text:
        return "breast_cancer"
    if any(k in text for k in ["colorectal", "colon", "rectal"]):
        return "colon_cancer"
    # Keep only the core classes for this hackathon prototype.
    return None


def load_geo_soft(path: str, keep_unmapped: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    samples: List[dict] = []
    meta_rows: List[dict] = []

    current_id = None
    title = ""
    source = ""
    values = {}
    collecting = False

    def flush():
        nonlocal current_id, title, source, values
        if current_id is None or not values:
            return
        label = infer_label_from_title(title, source)
        if label is None and not keep_unmapped:
            return
        row = {"sample_id": current_id, **values}
        samples.append(row)
        meta_rows.append({
            "sample_id": current_id,
            "title": title,
            "source": source,
            "label": label or "unmapped",
            "geo_file": os.path.basename(path),
        })

    with _open_text(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("^SAMPLE"):
                flush()
                current_id = line.split("=", 1)[1].strip()
                title, source, values, collecting = "", "", {}, False
                continue
            if current_id is None:
                continue
            if line.startswith("!Sample_title"):
                title = line.split("=", 1)[1].strip()
            elif line.startswith("!Sample_source_name"):
                source = line.split("=", 1)[1].strip()
            elif line.startswith("ID_REF"):
                collecting = True
            elif collecting:
                if not line.strip() or line.startswith("^") or line.startswith("!") or line.startswith("#"):
                    collecting = False
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    fid = str(parts[0]).strip()
                    try:
                        val = float(parts[1])
                    except ValueError:
                        val = np.nan
                    values[fid] = val
        flush()

    if not samples:
        raise ValueError(f"No usable labelled samples found in {path}")

    df = pd.DataFrame(samples).set_index("sample_id")
    meta = pd.DataFrame(meta_rows).set_index("sample_id")
    y = meta["label"].copy()
    X = df.apply(pd.to_numeric, errors="coerce")
    # Drop features missing in nearly all samples and constant features.
    X = X.dropna(axis=1, thresh=max(2, int(0.2 * len(X))))
    nunique = X.nunique(dropna=True)
    X = X.loc[:, nunique > 1]
    return X, y, meta.reset_index()


def load_multiple_geo_soft(paths: Iterable[str]) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    Xs, ys, metas = [], [], []
    for p in paths:
        X, y, meta = load_geo_soft(p)
        Xs.append(X)
        ys.append(y)
        metas.append(meta)
    Xall = pd.concat(Xs, axis=0, join="outer", sort=False)
    yall = pd.concat(ys, axis=0)
    meta_all = pd.concat(metas, axis=0, ignore_index=True)
    # Remove classes with very tiny support to avoid stratified split failures.
    counts = yall.value_counts()
    keep_labels = counts[counts >= 3].index
    keep = yall.isin(keep_labels)
    return Xall.loc[keep], yall.loc[keep], meta_all[meta_all["sample_id"].isin(yall.loc[keep].index)]


def convert_soft_to_csv(paths: Iterable[str], out_csv: str, out_meta_csv: Optional[str] = None) -> str:
    X, y, meta = load_multiple_geo_soft(paths)
    out = X.copy()
    out.insert(0, "label", y.values)
    out.insert(0, "sample_id", X.index)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    if out_meta_csv:
        meta.to_csv(out_meta_csv, index=False)
    return out_csv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--soft_paths", nargs="+", required=True)
    ap.add_argument("--out_csv", default="data/real_geo_mirna.csv")
    ap.add_argument("--out_meta_csv", default="data/real_geo_mirna_metadata.csv")
    args = ap.parse_args()
    convert_soft_to_csv(args.soft_paths, args.out_csv, args.out_meta_csv)
    print(f"Wrote {args.out_csv}")
