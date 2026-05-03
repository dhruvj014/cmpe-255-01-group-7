# CMPE 255-01: Data Mining Project — Group 7

## Team Members
- **Dhruv Sachin Jain** (SJSU ID: 019150859)
- **Disha Jadav** (SJSU ID: 018484362)
- **Himanshu Jain** (SJSU ID: 019098794)
- **Nitish Kumar** (SJSU ID: 019155916)

## Project Overview

**Fake Review Detection on Yelp Using Multi-Signal Analysis** on the YelpZip dataset (608,598 reviews, 13.2% spam rate).

A six-layer pipeline that fuses textual content, reviewer behavior, association rules, clustering structure, supervised classification, and synthetic-attack validation into a single detection stack.

| Layer | Component | Methods | Owner |
|-------|-----------|---------|-------|
| L1 | ETL / OLAP | PySpark ETL, OLAP cubes | Himanshu |
| L2 | Association Rules | FP-Growth on behavior baskets | Dhruv |
| L3 | Text Mining / LLM | DeBERTa-v1 fine-tuning | Nitish |
| L4 | Clustering | K-Means + DBSCAN | Dhruv |
| L5 | Classification / Anomaly | DT / RF / MLP / IsolationForest / LOF | Disha |
| L6 | Validation | Jaccard stability + synthetic attacks + ablation | Himanshu |

---

## Headline Results

**L3 standalone (DeBERTa-v1 on test split):**
- AUC-ROC **0.93**, Avg Precision **0.79**, F1-macro **0.84** (default threshold 0.5).

**L5 supervised (review-level holdout, 106,226 rows):**

| Model | AUC-ROC | F1@optimal | Avg Precision |
|-------|--------:|-----------:|--------------:|
| MLP | **0.944** | **0.741** | **0.807** |
| Random Forest | 0.941 | 0.722 | 0.781 |
| Decision Tree | 0.936 | 0.734 | 0.781 |

**L5 anomaly:** Isolation Forest 0.760 AUC, LOF 0.661 AUC.

**Ablation across layer subsets** (`L6_Validation/outputs/ablation_table.csv`):

| Configuration | AUC-ROC | F1@opt | AP |
|---|--:|--:|--:|
| L2-only | 0.716 | 0.385 | 0.230 |
| L4-only | 0.773 | 0.417 | 0.282 |
| L5-supervised (no L3) | 0.819 | 0.437 | 0.368 |
| L5-anomaly (no L3) | 0.572 | 0.254 | 0.176 |
| L2+L4 | 0.774 | 0.415 | 0.283 |
| L2+L4+L5 (full behavioral, no L3) | 0.815 | 0.438 | 0.365 |
| **Full + L3** | **0.936** | **0.741** | **0.798** |

L3 lifts the full stack from 0.815 → 0.936 AUC.

**L6 synthetic-attack detection (per-tier):**

| Layer | Easy | Medium | Hard |
|---|--:|--:|--:|
| L2 | 1.00 | 0.33 | 0.00 |
| L4-kmeans | 1.00 | 0.13 | 0.00 |
| L4-dbscan | 1.00 | 0.07 | 0.17 |
| **L5-supervised** | **1.00** | **1.00** | **0.33** |
| L5-anomaly | 0.93 | 0.80 | 0.05 |

Hard tier: 40 veteran-camouflaged synthetic profiles (expanded from 10 in earlier check-in).

---

## How to Run

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Per-layer execution

```bash
# L1 ETL (produces reviews_enriched.csv, reviewer_profiles.csv)
cd L1_ETL_OLAP && python3 main.py && cd ..

# L2 FP-Growth
python3 L2_FPGrowth/01_basket_encoding.py
python3 L2_FPGrowth/02_fpgrowth_mining.py
python3 L2_FPGrowth/03_rule_analysis.py

# L3 DeBERTa fine-tuning (requires GPU; see L3/AWS_TRAINING_GUIDE.md)
# Locally: predictions and metrics are checked into L3/outputs/.
# To refresh metrics from saved predictions:
python3 L3/scripts/refresh_metrics.py

# L4 Clustering
python3 L4_Clustering/01_preprocessing.py
python3 L4_Clustering/02_kmeans_clustering.py
python3 L4_Clustering/03_dbscan_clustering.py
python3 L4_Clustering/04_cluster_analysis.py

# L5 Classification + analyses
cd L5_Classification
python3 01_build_feature_table.py    # fuses L1 + L2 + L3 + L4 features
python3 02_train_models.py           # DT / RF / MLP supervised
python3 03_anomaly_detection.py      # Isolation Forest / LOF
python3 06_ensemble.py               # logistic stacker over all signals
python3 07_error_analysis.py         # FP/FN bucket analysis
python3 08_calibration.py            # Platt + isotonic calibration
cd ..

# L6 Validation
cd L6_Validation
python3 01_jaccard_stability.py
python3 02_synthetic_injection.py    # 70 synthetic profiles across 3 tiers
python3 03_summary_report.py
python3 04_ablation_study.py         # ablation across layer subsets
cd ..
```

### Tests

```bash
python3 -m pytest tests/ -v
```

---

## Repo Layout

- `L1_ETL_OLAP/` — Spark ETL, OLAP cubes, feature tables
- `L2_FPGrowth/` — FP-Growth association rule mining
- `L3/` — DeBERTa fine-tuning + saved predictions/metrics + helper scripts
- `L4_Clustering/` — K-Means + DBSCAN
- `L5_Classification/` — supervised + anomaly + ensemble + calibration + error analysis
- `L6_Validation/` — Jaccard stability + synthetic attacks + ablation
- `tools/` — figure-audit doc
- `tests/` — smoke tests for each layer's outputs
- `docs/` — design specs and implementation plans

---

## Dataset

YelpZip is not redistributed in this repo. Obtain from Mukherjee et al. and place the raw `.txt` files under `L1_ETL_OLAP/data/`.
