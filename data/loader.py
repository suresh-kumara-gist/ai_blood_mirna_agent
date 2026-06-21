import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from typing import Optional, Tuple, Dict


def load_expression_csv(path_or_file, label_col: Optional[str] = "label") -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Load GEO/TCGA-style CSV.

    Supported shapes:
    1. samples x miRNAs, with optional label column.
    2. genes/miRNAs x samples where first column is miRNA id. Use transpose_expression_matrix.
    """
    df = pd.read_csv(path_or_file)
    labels = None
    if label_col and label_col in df.columns:
        labels = df[label_col].astype(str)
        df = df.drop(columns=[label_col])
    df = df.select_dtypes(include=[np.number]).copy()
    return df, labels


def transpose_expression_matrix(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    mat = df.set_index(id_col).T
    mat.index.name = "sample_id"
    return mat.reset_index(drop=True)


def preprocess_train(X: pd.DataFrame, y: pd.Series, max_features: int = 512) -> Tuple[np.ndarray, np.ndarray, Dict]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    # Keep training feasible for high-dimensional GEO arrays. Select features before scaling.
    X_imp = imputer.fit_transform(X)
    selector = None
    selected_feature_names = list(X.columns)
    if max_features and X_imp.shape[1] > max_features:
        selector = SelectKBest(score_func=f_classif, k=max_features)
        X_imp = selector.fit_transform(X_imp, y_enc)
        selected_feature_names = list(np.array(X.columns)[selector.get_support()])
    X_scaled = scaler.fit_transform(X_imp)
    meta = {
        "feature_names": list(X.columns),
        "selected_feature_names": selected_feature_names,
        "selector": selector,
        "imputer": imputer,
        "scaler": scaler,
        "label_encoder": encoder,
    }
    return X_scaled.astype("float32"), y_enc.astype("int64"), meta


def preprocess_inference(X: pd.DataFrame, meta: Dict) -> np.ndarray:
    feature_names = meta["feature_names"]
    for col in feature_names:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_names]
    X_imp = meta["imputer"].transform(X)
    selector = meta.get("selector")
    if selector is not None:
        X_imp = selector.transform(X_imp)
    X_scaled = meta["scaler"].transform(X_imp)
    return X_scaled.astype("float32")


def augment_data(X: np.ndarray, y: np.ndarray, copies: int = 2, noise_std: float = 0.05, mask_prob: float = 0.03) -> Tuple[np.ndarray, np.ndarray]:
    """Hybrid augmentation for real + synthetic data."""
    rng = np.random.default_rng(42)
    Xs, ys = [X], [y]
    for _ in range(copies):
        Xa = X.copy()
        Xa += rng.normal(0, noise_std, Xa.shape)
        Xa *= rng.normal(1.0, 0.03, size=(Xa.shape[0], 1))
        mask = rng.random(Xa.shape) < mask_prob
        Xa[mask] = 0.0
        Xs.append(Xa.astype("float32"))
        ys.append(y)
    return np.vstack(Xs), np.concatenate(ys)
