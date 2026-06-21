import numpy as np
import pandas as pd

from model.inference import load_model, predict_samples
from agent.llm_report_generator import generate_report


class AIBloodMiRNADiagnosticAgent:
    def __init__(self, artifact_dir="artifacts"):
        self.artifact_dir = artifact_dir
        self.model, self.meta = load_model(artifact_dir)

    def run(self, csv_file, sample_index=0, explain=True):
        csv_file.seek(0)
        df = pd.read_csv(csv_file)

        raw = predict_samples(
            Xdf=df,
            model=self.model,
            meta=self.meta,
        )

        probabilities = np.asarray(raw["probabilities"])[sample_index]
        class_names = raw["class_names"]
        uncertainty = float(np.asarray(raw["uncertainty"])[sample_index])
        anomaly_score = float(np.asarray(raw["anomaly_score"])[sample_index])
        predicted_label = raw["predicted_labels"][sample_index]

        top_features = None
        fig = None

        try:
            report = generate_report(
                predicted_label,
                probabilities,
                class_names,
                uncertainty,
                anomaly_score,
                top_features,
            )
        except Exception:
            confidence = float(np.max(probabilities))
            report = f"""
AI Blood microRNA Multi-Cancer Detection Agent — Research Use Only

Predicted pattern: {predicted_label}
Confidence score: {confidence:.2%}
Uncertainty score: {uncertainty:.4f}
Anomaly score: {anomaly_score:.4f}

This is not a diagnosis. This result is a research-stage AI risk interpretation based on the uploaded microRNA expression profile.

Recommendation:
Please review this result with a qualified clinician and correlate with clinical history, imaging, laboratory tests, and confirmatory diagnostics.
"""

        return {
            "raw": raw,
            "top_features": top_features,
            "figure": fig,
            "report": report,
        }

# import pandas as pd

# from model.inference import load_model, predict_samples
# from agent.llm_report_generator import generate_report


# class AIBloodMiRNADiagnosticAgent:
#     def __init__(self, artifact_dir="artifacts"):
#         self.artifact_dir = artifact_dir
#         self.model, self.meta = load_model(artifact_dir)

#     def run(self, csv_file, sample_index=0, explain=True):
#         csv_file.seek(0)
#         df = pd.read_csv(csv_file)

#         raw = predict_samples(
#             Xdf=df,
#             model=self.model,
#             meta=self.meta,
#         )

#         try:
#             report = generate_report(
#                 raw,
#                 top_features=None,
#                 sample_index=sample_index,
#             )
#         except TypeError:
#             try:
#                 report = generate_report(raw, None)
#             except TypeError:
#                 report = generate_report(raw)

#         return {
#             "raw": raw,
#             "top_features": None,
#             "figure": None,
#             "report": report,
#         }

# import pandas as pd

# from model.inference import load_model, predict_samples
# from agent.llm_report_generator import generate_report


# class AIBloodMiRNADiagnosticAgent:
#     def __init__(self, artifact_dir="artifacts"):
#         self.artifact_dir = artifact_dir
#         self.model, self.meta = load_model(artifact_dir)

#     def run(self, csv_file, sample_index=0, explain=True):
#         csv_file.seek(0)
#         df = pd.read_csv(csv_file)

#         raw = predict_samples(
#             Xdf=df,
#             model=self.model,
#             meta=self.meta,
#         )

#         report = generate_report(
#             raw,
#             top_features=None,
#             sample_index=sample_index,
#         )

#         return {
#             "raw": raw,
#             "top_features": None,
#             "figure": None,
#             "report": report,
#         }

# import os
# import sys

# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# if BASE_DIR not in sys.path:
#     sys.path.insert(0, BASE_DIR)

# import pandas as pd

# from model.inference import load_model, predict_samples, MiRNAInferencePipeline
# from explainability.shap_explainer import explain_with_shap
# from agent.llm_report_generator import generate_report


# class AIBloodMiRNADiagnosticAgent:
#     def __init__(self, artifact_dir="artifacts"):
#         self.artifact_dir = artifact_dir
#         self.model, self.meta = load_model(artifact_dir)
#         self.pipeline = MiRNAInferencePipeline(artifact_dir)

#     def run(self, csv_file, sample_index=0, explain=True):
#         if hasattr(csv_file, "seek"):
#             csv_file.seek(0)
#         Xdf = pd.read_csv(csv_file)
#         Xdf = Xdf.drop(columns=["sample_id", "label"], errors="ignore")

#         result = predict_samples(Xdf, self.model, self.meta)

#         top_features, fig = None, None
#         if explain:
#             try:
#                 top_features, fig = explain_with_shap(
#                     model=self.model,
#                     X_preprocessed=result["X_preprocessed"],
#                     feature_names=result.get("model_feature_names"),
#                     sample_index=sample_index,
#                 )
#             except Exception:
#                 # Prediction should never fail because explainability failed.
#                 top_features, fig = None, None

#         report = generate_report(
#             predicted_label=result["predicted_labels"][sample_index],
#             probabilities=result["probabilities"][sample_index],
#             class_names=result["class_names"],
#             uncertainty=float(result["uncertainty"][sample_index]),
#             anomaly_score=float(result["anomaly_score"][sample_index]),
#             top_features=top_features,
#         )

#         return {
#             "raw": result,
#             "top_features": top_features,
#             "figure": fig,
#             "report": report,
#         }
