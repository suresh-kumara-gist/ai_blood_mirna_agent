import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def _safe_1d(values, probs=None, sample_index=0):
    class_idx = 0
    if probs is not None:
        probs = np.asarray(probs)
        if probs.ndim == 2:
            class_idx = int(np.argmax(probs[0]))
        elif probs.ndim == 1:
            class_idx = int(np.argmax(probs))

    if isinstance(values, list):
        values = values[min(class_idx, len(values) - 1)]

    arr = np.asarray(values)
    arr = np.squeeze(arr)

    if arr.ndim == 1:
        return arr.astype(float)
    if arr.ndim == 2:
        return arr[min(sample_index, arr.shape[0] - 1)].astype(float)
    if arr.ndim == 3:
        # samples x features x classes
        if probs is not None and arr.shape[-1] == np.asarray(probs).shape[-1]:
            return arr[min(sample_index, arr.shape[0] - 1), :, class_idx].astype(float)
        # classes x samples x features
        if probs is not None and arr.shape[0] == np.asarray(probs).shape[-1]:
            return arr[class_idx, min(sample_index, arr.shape[1] - 1), :].astype(float)
        return arr.reshape(arr.shape[0], -1)[min(sample_index, arr.shape[0] - 1)].astype(float)

    return arr.reshape(-1).astype(float)


def _fallback_explanation(X_preprocessed, feature_names, sample_index=0):
    X = np.asarray(X_preprocessed, dtype=np.float32)
    sample = X[min(sample_index, len(X) - 1)]
    if len(X) > 1:
        baseline = X.mean(axis=0)
        vals = sample - baseline
    else:
        vals = sample
    return vals[: len(feature_names)]


def explain_with_shap(
    model,
    X_preprocessed,
    feature_names=None,
    sample_index: int = 0,
    max_background: int = 50,
    out_path: str = None,
    **kwargs,
):
    """Robust explanation: tries SHAP; falls back to feature deviation.

    Supports old and new calls. Never raises dataframe dimensionality errors.
    """
    X_preprocessed = np.asarray(X_preprocessed, dtype=np.float32)
    if X_preprocessed.ndim == 1:
        X_preprocessed = X_preprocessed.reshape(1, -1)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_preprocessed.shape[1])]
    feature_names = list(feature_names)

    vals = None
    probs = None

    try:
        import shap

        def f(x):
            model.eval()
            x = np.asarray(x, dtype=np.float32)
            if x.ndim == 1:
                x = x.reshape(1, -1)
            with torch.no_grad():
                logits = model(torch.tensor(x, dtype=torch.float32))
                return torch.softmax(logits, dim=1).cpu().numpy()

        sample = X_preprocessed[[min(sample_index, len(X_preprocessed) - 1)]]
        background = X_preprocessed[: max(1, min(max_background, len(X_preprocessed)))]
        probs = f(sample)
        explainer = shap.KernelExplainer(f, background)
        shap_values = explainer.shap_values(sample, nsamples=100)
        vals = _safe_1d(shap_values, probs=probs, sample_index=0)
    except Exception:
        vals = _fallback_explanation(X_preprocessed, feature_names, sample_index=sample_index)

    vals = np.asarray(vals, dtype=float).reshape(-1)
    n = min(len(vals), len(feature_names))
    vals = vals[:n]
    feature_names = feature_names[:n]

    if n == 0:
        top = pd.DataFrame(columns=["microRNA", "importance", "direction"])
        return top, None

    order = np.argsort(np.abs(vals))[::-1][: min(10, n)]
    selected_names = [feature_names[i] for i in order]
    selected_vals = vals[order].astype(float).reshape(-1)

    top = pd.DataFrame({
        "microRNA": selected_names,
        "importance": selected_vals,
        "direction": ["increases risk" if v > 0 else "decreases risk" for v in selected_vals],
    })

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top["microRNA"][::-1], top["importance"][::-1])
    ax.set_title("Top microRNA contributors")
    ax.set_xlabel("SHAP / relative contribution")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
    return top, fig
