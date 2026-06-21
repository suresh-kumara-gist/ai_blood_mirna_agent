import os
import joblib
import torch
import numpy as np
import pandas as pd

from data.loader import preprocess_inference
from model.transformer import TabularMiRNATransformer


def _build_model(n_features, n_classes):
    try:
        return TabularMiRNATransformer(n_features, n_classes)
    except TypeError:
        try:
            return TabularMiRNATransformer(
                n_features=n_features,
                n_classes=n_classes,
            )
        except TypeError:
            try:
                return TabularMiRNATransformer(
                    num_features=n_features,
                    num_classes=n_classes,
                )
            except TypeError:
                return TabularMiRNATransformer(
                    input_size=n_features,
                    output_size=n_classes,
                )


def load_model(artifact_dir="artifacts"):
    meta_path = os.path.join(artifact_dir, "preprocess_meta.joblib")
    model_path = os.path.join(artifact_dir, "mirna_transformer.pt")

    meta = joblib.load(meta_path)

    model_feature_names = meta.get(
        "selected_feature_names",
        meta.get("feature_names"),
    )

    if model_feature_names is None:
        raise ValueError("No feature names found in preprocess metadata.")

    n_features = len(model_feature_names)
    n_classes = len(meta["label_encoder"].classes_)

    model = _build_model(n_features, n_classes)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, meta


def predict_samples(Xdf: pd.DataFrame, model, meta, mc_dropout_passes: int = 20):
    X = preprocess_inference(Xdf.copy(), meta)

    X = np.asarray(X, dtype=np.float32)
    xb = torch.tensor(X, dtype=torch.float32)

    probs_runs = []

    model.train()

    with torch.no_grad():
        for _ in range(mc_dropout_passes):
            logits = model(xb)
            probs_runs.append(torch.softmax(logits, dim=1).cpu().numpy())

    model.eval()

    probs_stack = np.stack(probs_runs)
    probs = probs_stack.mean(axis=0)

    uncertainty = probs_stack.std(axis=0).mean(axis=1)

    anomaly = np.mean(np.abs(X), axis=1) / (
        np.std(np.abs(X), axis=1) + 1e-6
    )

    labels = meta["label_encoder"].inverse_transform(
        probs.argmax(axis=1)
    )

    return {
        "probabilities": probs,
        "predicted_labels": labels,
        "uncertainty": uncertainty,
        "anomaly_score": anomaly,
        "class_names": list(meta["label_encoder"].classes_),
        "X_preprocessed": X,
        "model_feature_names": meta.get(
            "selected_feature_names",
            meta.get("feature_names"),
        ),
    }

# import os
# import sys

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if BASE_DIR not in sys.path:
#     sys.path.insert(0, BASE_DIR)

# import joblib
# import torch
# import numpy as np
# import pandas as pd

# from data.loader import preprocess_inference
# from model.transformer import TabularMiRNATransformer


# def load_model(artifact_dir="artifacts"):
#     meta_path = os.path.join(artifact_dir, "preprocess_meta.joblib")
#     model_path = os.path.join(artifact_dir, "mirna_transformer.pt")

#     if not os.path.exists(meta_path):
#         raise FileNotFoundError(f"Missing preprocessing metadata: {meta_path}")
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Missing trained model: {model_path}")

#     meta = joblib.load(meta_path)
#     model_feature_names = meta.get("selected_feature_names", meta.get("feature_names"))
#     if not model_feature_names:
#         raise ValueError("Metadata missing feature names.")

#     model = TabularMiRNATransformer(
#         input_dim=len(model_feature_names),
#         num_classes=len(meta["label_encoder"].classes_),
#     )
#     model.load_state_dict(torch.load(model_path, map_location="cpu"))
#     model.eval()
#     return model, meta


# def _prepare_input_df(path_or_file):
#     df = pd.read_csv(path_or_file)
#     return df.drop(columns=["sample_id", "label"], errors="ignore")


# def predict_samples(Xdf: pd.DataFrame, model, meta, mc_dropout_passes: int = 20):
#     X = preprocess_inference(Xdf.copy(), meta)
#     xb = torch.tensor(X, dtype=torch.float32)

#     probs_runs = []
#     model.train()  # enables dropout for MC uncertainty if dropout exists
#     with torch.no_grad():
#         passes = max(int(mc_dropout_passes), 1)
#         for _ in range(passes):
#             logits = model(xb)
#             probs_runs.append(torch.softmax(logits, dim=1).cpu().numpy())

#     model.eval()
#     probs_stack = np.stack(probs_runs)
#     probs = probs_stack.mean(axis=0)
#     uncertainty = probs_stack.std(axis=0).mean(axis=1)

#     anomaly = np.mean(np.abs(X), axis=1) / (np.std(np.abs(X), axis=1) + 1e-6)
#     pred_idx = probs.argmax(axis=1)
#     labels = meta["label_encoder"].inverse_transform(pred_idx)

#     return {
#         "probabilities": probs,
#         "predicted_labels": list(labels),
#         "predictions": pred_idx.tolist(),
#         "uncertainty": uncertainty.tolist(),
#         "anomaly_score": anomaly.tolist(),
#         "class_names": list(meta["label_encoder"].classes_),
#         "X_preprocessed": X,
#         "model_feature_names": meta.get("selected_feature_names", meta.get("feature_names")),
#     }


# class MiRNAInferencePipeline:
#     def __init__(self, artifact_dir="artifacts"):
#         self.artifact_dir = artifact_dir
#         self.model, self.meta = load_model(artifact_dir)

#     def predict(self, csv_file, mc_dropout_passes: int = 20):
#         Xdf = _prepare_input_df(csv_file)
#         return predict_samples(Xdf, self.model, self.meta, mc_dropout_passes=mc_dropout_passes)


# # Backward-compatible alias.
# MiRNAInference = MiRNAInferencePipeline
