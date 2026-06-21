SYSTEM_PROMPT = """
You are a clinical AI assistant. Convert structured ML outputs into a clear,
non-alarming medical-style report. Do not give final diagnosis. Only provide
risk interpretation and recommendations. Mention that this is research/prototype
software and must be reviewed by qualified clinicians.
"""


def generate_report(predicted_label, probabilities, class_names, uncertainty, anomaly_score, top_features):
    top = ", ".join(top_features["microRNA"].head(5).tolist()) if top_features is not None else "not available"
    prob_map = {c: round(float(p) * 100, 2) for c, p in zip(class_names, probabilities)}
    max_prob = max(prob_map.values())
    uncertainty_note = "The result is flagged as inconclusive; repeat testing or additional clinical review is recommended." if uncertainty > 0.08 else "Model uncertainty is not elevated for this sample."
    return f"""{SYSTEM_PROMPT.strip()}

Clinical-style risk interpretation:
The microRNA expression profile shows the highest similarity to: {predicted_label}.
Estimated confidence for this class: {max_prob:.2f}%.
Cancer probability distribution: {prob_map}.
Uncertainty score: {uncertainty:.4f}. Anomaly score: {anomaly_score:.4f}.

Key contributing microRNAs include: {top}.
{uncertainty_note}

Important: This output is not a diagnosis. It is a prototype AI risk-screening report and should only be used for research, education, or hackathon demonstration. Confirmatory laboratory tests and clinician review are required.
"""
