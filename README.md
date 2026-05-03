# mlBased-NIDS-research

> Cross-dataset generalization study of ML-based Intrusion Detection Systems
> with protocol-level analysis of dataset shift.
> **Course:** Network Protocols & Security (NPS)

---

## Paper Title

**"Dataset Shift in ML-Based Intrusion Detection Systems: A Protocol-Level Analysis of Generalization Failure"**

---

## Research Question

> "Do ML models trained on NSL-KDD generalize to CICIDS 2017, and can the performance gap be explained at the network protocol level?"

---

## Project Structure

```
IDS/
├── data/
│   ├── raw/
│   │   ├── KDDTrain+.arff
│   │   ├── KDDTest+.arff
│   │   └── MachineLearningCSV/
│   │       └── MachineLearningCVE/     ← 8 CICIDS daily CSV files
│   └── processed/
│       ├── cicids_combined.csv         ← 10% stratified sample of all days
│       ├── cicids_labels.csv           ← original CICIDS labels (aligned to cleaned rows)
│       ├── kdd_train_control.csv
│       ├── kdd_test_control.csv
│       ├── cicids_control.csv
│       ├── kdd_train_protocol_aware.csv
│       ├── kdd_test_protocol_aware.csv
│       ├── cicids_protocol_aware.csv
│       ├── kdd_train_exp[1-5].csv
│       ├── kdd_test_exp[1-5].csv
│       └── cicids_exp[1-5].csv
├── models/                             ← saved .pkl (model + scaler + features + numeric_cols)
│   ├── lr_exp[0-5].pkl
│   ├── rf_exp[0-5].pkl
│   └── svm_exp[0-5].pkl
├── results/
│   ├── feature_importance_all.csv      ← combined feature importance all models all experiments
│   ├── attack_breakdown_all.csv        ← per-attack-type catch rate all models all experiments
│   └── experiment_results.csv          ← F1/Recall/Precision summary all models all experiments
├── src/
│   ├── loader.py
│   ├── cleaner.py
│   ├── extractor.py
│   ├── aligner.py
│   ├── preprocessor.py
│   ├── pipelines.py
│   ├── models/
│   │   ├── logistic_regression.py
│   │   ├── random_forest.py
│   │   └── svm.py
│   └── experiments/
│       ├── feature_importance.py
│       ├── attack_breakdown.py
│       └── visualizations.py
├── tests/
│   ├── test_loader.py
│   └── test_cleaner.py
├── scripts/
│   └── combine_cicids.py
├── config.py
└── README.md
```

---

## Setup

```bash
git clone https://github.com/yourname/mlBased-NIDS-research.git
cd mlBased-NIDS-research

uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

uv pip install pandas numpy scipy scikit-learn matplotlib seaborn flask joblib pytest ipykernel
```

---

## How To Run

### Step 1 — Prepare CICIDS Dataset
```bash
python scripts/combine_cicids.py
```

### Step 2 — Build All Experiment Datasets
```bash
python -c "from src.pipelines import run_control_pipeline, run_protocol_aware_pipeline, run_all_experiment_pipelines; run_control_pipeline(); run_protocol_aware_pipeline(); run_all_experiment_pipelines()"
```

### Step 3 — Run Models
```bash
python -c "from src.models.logistic_regression import *; [run_logistic_regression_EXP0(), run_logistic_regression_EXP1(), run_logistic_regression_EXP2(), run_logistic_regression_EXP3(), run_logistic_regression_EXP4(), run_logistic_regression_EXP5()]"
python -c "from src.models.random_forest import *; [run_random_forest_EXP0(), run_random_forest_EXP1(), run_random_forest_EXP2(), run_random_forest_EXP3(), run_random_forest_EXP4(), run_random_forest_EXP5()]"
python -c "from src.models.svm import *; [run_svm_EXP0(), run_svm_EXP1(), run_svm_EXP2(), run_svm_EXP3(), run_svm_EXP4(), run_svm_EXP5()]"
```

### Step 4 — Run Analysis
```bash
python src/experiments/feature_importance.py
python src/experiments/attack_breakdown.py
```

### Step 5 — Run Tests
```bash
pytest tests/ -v -s
```

---

## Datasets

| Dataset | Year | Environment | Collection | Features | Labels |
|---|---|---|---|---|---|
| NSL-KDD | 1999/2009 | Simulated lab | Per-CONNECTION, manual | 41 | normal, DoS, Probe, R2L, U2R |
| CICIDS 2017 | 2017 | Realistic modern network | Per-FLOW, CICFlowMeter | 78 | BENIGN + 14 attack types |

---

## Experiment Design

### Feature Sets

| Exp | Change | Features |
|---|---|---|
| EXP0 | Control (baseline) | duration, src_bytes, dst_bytes, bytes_per_sec, byte_ratio, protocol_* |
| EXP1 | Swap src_bytes → syn_ratio | duration, syn_ratio, dst_bytes, bytes_per_sec, byte_ratio, protocol_* |
| EXP2 | Swap dst_bytes → rst_ratio | duration, syn_ratio, rst_ratio, bytes_per_sec, byte_ratio, protocol_* |
| EXP3 | Add fin_ratio | EXP2 + fin_ratio |
| EXP4 | Add data_pkt_ratio | EXP3 + data_pkt_ratio |
| EXP5 | Add service_bucket | EXP4 + svc_http, svc_ssh, svc_ftp, svc_dns... |

### Protocol-Aware Feature Mapping

| Feature | KDD Source | CICIDS Source | Protocol Meaning |
|---|---|---|---|
| `syn_ratio` | `serror_rate` | `SYN Flag Count / Total Fwd Packets` | SYN flood intensity |
| `rst_ratio` | `rerror_rate` | `RST Flag Count / Total Fwd Packets` | Port scan / closed port hits |
| `fin_ratio` | `flag == "SF"` | `FIN Flag Count / Total Fwd Packets` | Connection lifecycle completion |
| `data_pkt_ratio` | `src_bytes > 0` | `act_data_pkt_fwd / Total Fwd Packets` | Payload presence |
| `service_bucket` | `map_service_to_bucket(service)` | `port_bucket(Destination Port)` | IANA protocol tier |

---

## Results

### Complete Results Table

| Exp | Features | LR Within | LR Cross | LR Drop | RF Within | RF Cross | RF Drop | SVM Within | SVM Cross | SVM Drop |
|---|---|---|---|---|---|---|---|---|---|---|
| EXP0 | Control | 0.7363 | 0.2312 | 0.5051 | 0.7991 | 0.2621 | 0.5370 | 0.7362 | **0.3011** | **0.4351** |
| EXP1 | +syn_ratio | 0.7387 | 0.2428 | 0.4959 | 0.7694 | 0.2617 | 0.5076 | 0.7473 | 0.2312 | 0.5161 |
| EXP2 | +rst_ratio | 0.7771 | 0.0204 | 0.7567 | 0.8041 | 0.1496 | 0.6546 | 0.7881 | 0.0201 | 0.7680 |
| EXP3 | +fin_ratio | 0.7769 | 0.0181 | 0.7588 | 0.7929 | 0.0792 | 0.7137 | 0.7883 | 0.0183 | 0.7701 |
| EXP4 | +data_pkt | **0.8204** | **0.3105** | 0.5099 | 0.7961 | 0.1541 | 0.6420 | 0.7896 | 0.0210 | 0.7686 |
| EXP5 | +service | 0.7466 | 0.1938 | 0.5527 | **0.8247** | 0.1519 | 0.6728 | 0.7617 | 0.2198 | 0.5419 |

### Key Findings

**Finding 1 — Protocol features consistently worsen generalization**
Best cross-dataset F1 across all 18 experiments is SVM EXP0 (0.3011) — the control baseline with no protocol features.

**Finding 2 — rst_ratio is the most damaging single feature**
Adding rst_ratio (EXP2) causes the largest single-step drop across all three models. RST-based scanning disappeared from CICIDS 2017 — modern attacks use HTTP floods that generate no RST traffic.

**Finding 3 — The SVM paradox**
SVM achieves best cross-dataset F1 without protocol features (0.3011) but collapses to 0.0201 when rst_ratio is added. Maximum margin commits to a boundary that becomes directionally reversed in CICIDS.

**Finding 4 — syn_ratio is the only protocol feature that doesn't hurt**
Adding syn_ratio causes minimal damage and slightly improves LR. syn_ratio is near zero for both classes in CICIDS — adds minimal noise in either direction.

**Finding 5 — LR EXP4 is the unexpected best performer**
LR with data_pkt_ratio achieves 0.3105 cross-dataset F1 — highest of any experiment. data_pkt_ratio partially distinguishes Slowloris-style attacks via payload presence.

**Finding 6 — Within-dataset F1 is a misleading metric**
RF EXP5 has highest within-dataset F1 (0.8247) but poor cross-dataset F1 (0.1519). The best benchmark model is consistently not the best real-world performer.

---

## What Has Been Done

### Data Pipeline
- [x] NSL-KDD loaded from `.arff`, byte strings decoded, converted to CSV
- [x] CICIDS 2017 — all 8 daily files combined, stratified 10% sample with minimum 10 rows per class
- [x] Cleaning — nulls, duplicates, infinities removed from both datasets
- [x] `cicids_labels.csv` saved at cleaning step for row-aligned per-attack analysis
- [x] Control feature extraction (protocol-unaware: duration, src_bytes, dst_bytes, bytes_per_sec, byte_ratio, protocol)
- [x] Protocol-aware feature extraction (syn_ratio, rst_ratio, fin_ratio, data_pkt_ratio, service_bucket)
- [x] log1p applied to volume features only — ratio columns explicitly excluded
- [x] StandardScaler fitted on KDD train only, applied to KDD test and CICIDS
- [x] 6 experiment CSVs generated (EXP0-EXP5) for all 3 datasets

### Models
- [x] Logistic Regression — EXP0 through EXP5
- [x] Random Forest — EXP0 through EXP5
- [x] SVM (LinearSVC) — EXP0 through EXP5
- [x] All models saved as `.pkl` with model + scaler + feature_cols + numeric_cols

### Analysis
- [x] Feature importance — coefficients (LR/SVM) and importances (RF) per experiment
- [x] Per-attack-type breakdown — catch rate per CICIDS attack class per model per experiment
- [x] Combined results CSVs saved to `results/`

### Testing
- [x] test_loader.py
- [x] test_cleaner.py

---

## What Still Needs To Be Done

### Visualizations
- [ ] Cross-dataset F1 drop per experiment (line chart — all 3 models)
- [ ] Per-attack-type catch rate heatmap (attack type vs experiment)
- [ ] Feature importance bar charts (top features per model EXP0 vs EXP2)
- [ ] Within vs cross F1 comparison (grouped bar chart)

### Paper
- [ ] Abstract
- [ ] Introduction
- [ ] Background
- [ ] Protocol Vulnerability Analysis (TCP, HTTP, DNS)
- [ ] Methodology
- [ ] Results
- [ ] Protocol-Level Analysis
- [ ] Discussion + Conclusion

---

## Core Argument

Protocol-aware ML features improve within-dataset IDS performance but worsen cross-dataset generalization. The 14-year gap between NSL-KDD (2003) and CICIDS 2017 represents not just statistical drift but a fundamental migration of attack behavior from Layer 3/4 protocol grammar violations to Layer 7 application semantics. Features grounded in attack techniques (SYN floods, RST-based scanning) fail catastrophically when those techniques evolve. No protocol-aware feature combination outperforms the simple control baseline on cross-dataset F1.

---

*Last updated: After completion of all analysis — feature importance, attack breakdown, full results table*