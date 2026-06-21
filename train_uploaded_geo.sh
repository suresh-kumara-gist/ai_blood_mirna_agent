#!/usr/bin/env bash
set -euo pipefail
mkdir -p artifacts
python -m model.train \
  --soft_paths \
  "data/raw/GSE74190_family_Lung_cancer_miRNA dataset.soft.gz" \
  "data/raw/GSE106817_family_Pan_cancer_miRNA.soft.gz" \
  "data/raw/GSE45666_family_Breast_cancer_miRNA_dataset.soft.gz" \
  --epochs "${EPOCHS:-12}" \
  --max_features "${MAX_FEATURES:-512}" \
  --augment \
  --out_dir artifacts
