# Training on uploaded GEO SOFT microRNA files

Copy your `.soft.gz` files into `data/raw/`, then run:

```bash
docker compose run --rm train-real
```

Or run manually:

```bash
docker compose run --rm app python -m model.train \
  --soft_paths "data/raw/GSE74190_family_Lung_cancer_miRNA dataset.soft.gz" \
               "data/raw/GSE106817_family_Pan_cancer_miRNA.soft.gz" \
               "data/raw/GSE45666_family_Breast_cancer_miRNA_dataset.soft.gz" \
  --epochs 20 --augment --out_dir artifacts
```

The loader maps labels to four hackathon classes:

- `normal`
- `lung_cancer`
- `breast_cancer`
- `colon_cancer`

It excludes ambiguous benign/borderline/unrelated cancer samples by default.

You can also convert SOFT to CSV first:

```bash
docker compose run --rm app python -m data.geo_soft_loader \
  --soft_paths data/raw/*.soft.gz \
  --out_csv data/real_geo_mirna.csv \
  --out_meta_csv data/real_geo_mirna_metadata.csv
```

## Fast first run on Intel Mac

```bash
MAX_FEATURES=256 EPOCHS=5 docker compose run --rm train-real
```

For better accuracy after the first successful run:

```bash
MAX_FEATURES=512 EPOCHS=20 docker compose run --rm train-real
```

Saved outputs:

- `artifacts/mirna_transformer.pt`
- `artifacts/preprocess_meta.joblib`
