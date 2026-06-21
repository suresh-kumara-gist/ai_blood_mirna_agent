# 🧬 AI Blood microRNA Multi-Cancer Detection Agent

prototype for blood microRNA expression based early cancer risk screening.

> ⚠️ Not a medical device. Not for diagnosis. Research/demo use only.

## Features

- Real GEO/TCGA-style CSV loader
- Synthetic microRNA dataset generator
- Hybrid augmentation: Gaussian noise, masking, scaling
- PyTorch tabular Transformer model
- Multi-class classification: normal, lung, breast, colon cancer
- Outputs cancer probability, predicted type, uncertainty, anomaly score
- SHAP explainability with top 10 microRNAs
- AI diagnostic agent that generates clinical-style reports
- Streamlit dashboard

## Project Structure

```text
project/
├── data/
│   ├── synthetic_generator.py
│   ├── loader.py
├── model/
│   ├── transformer.py
│   ├── train.py
│   ├── inference.py
├── explainability/
│   ├── shap_explainer.py
├── agent/
│   ├── diagnostic_agent.py
│   ├── llm_report_generator.py
├── app/
│   ├── streamlit_app.py
├── artifacts/
├── requirements.txt
├── README.md
```

## Setup

```bash
cd ai_blood_mirna_agent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Generate Synthetic Dataset

```bash
python data/synthetic_generator.py
```

## Train Model

```bash
python -m model.train --epochs 15 --augment --out_dir artifacts
```

For your own CSV:

```bash
python -m model.train --csv_path path/to/geo_tcga_mirna.csv --label_col label --epochs 20 --augment
```

CSV format expected:

```csv
miR-21,miR-155,miR-34a,...,label
9.1,7.2,5.4,...,lung_cancer
```

Labels supported by synthetic generator:

- normal
- lung_cancer
- breast_cancer
- colon_cancer

## Run Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

Upload a CSV with the same microRNA feature columns used during training.

## Agent Example

```python
from agent.diagnostic_agent import AIBloodMiRNADiagnosticAgent

agent = AIBloodMiRNADiagnosticAgent(artifact_dir="artifacts")
result = agent.run("patient_sample.csv")
print(result["report"])
```

## Clinical Report Prompt

The report generator uses this safety prompt:

```text
You are a clinical AI assistant. Convert structured ML outputs into a clear,
non-alarming medical-style report. Do not give final diagnosis. Only provide
risk interpretation and recommendations.
```

## Production Notes

Before any real clinical use, you would need:

- IRB/ethics-reviewed datasets
- External validation cohorts
- Batch-effect correction
- Proper calibration and confidence intervals
- Regulatory pathway review
- Human clinician oversight
- Secure PHI handling


---

## Docker Compose Setup

Build and run the Streamlit dashboard:

```bash
docker compose up --build
```

Open `http://localhost:8501`.

Train the demo model either from the Streamlit sidebar or from CLI:

```bash
docker compose run --rm train-demo
```

Persistent folders:

- `./artifacts` stores model checkpoints and preprocessing metadata.
- `./data` stores generated or uploaded CSV files.

Stop containers:

```bash
docker compose down
```
