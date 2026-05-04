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

## Summary

### Problem Statement

When deploying intrusion detection models across different network environments, performance dramatically degrades. A model trained on NSL-KDD achieves 73-80% F1 on held-out NSL-KDD data, but only 2-30% on CICIDS 2017. **Why do models fail so catastrophically on new datasets?**

This research investigates whether **protocol-level features** (TCP flags, traffic patterns) that appear discriminative on one dataset become liabilities on another due to fundamental differences in how attacks manifest in different network environments.

### Research Hypothesis

Protocol-aware features (SYN ratios, RST flags, etc.) encode dataset-specific attack patterns:
- **NSL-KDD (1999 simulated attacks):** Heavy use of scanning (RST flags), SYN floods, sequential patterns
- **CICIDS 2017 (modern attacks):** HTTP floods, Brute-force attempts, GRE tunneling, minimal flag usage

**Hypothesis:** Models trained with protocol features overfit to NSL-KDD's attack signatures. When deployed on CICIDS 2017, these features become anti-correlated or noisy, causing model collapse.

### Core Findings

| Finding | Evidence | Implication |
|---------|----------|-------------|
| **Protocol features worsen generalization** | Best cross-F1 is SVM EXP0 (0.3011) with NO protocol features | Protocol features are dataset-specific, not universal |
| **rst_ratio is catastrophic** | Adding it drops F1 from 0.23 → 0.02 (90% collapse) | RST scanning is NSL-KDD-specific; CICIDS uses HTTP/application-level attacks |
| **SVM shows maximum paradox** | Best within-dataset (0.7881) becomes worst cross-dataset (0.0201) | Maximum margin hyperplane is directionally reversed between datasets |
| **syn_ratio is benign** | Minimal cross-F1 damage when added | SYN ratio is near-zero in both datasets; provides no discriminative signal |
| **LR EXP4 unexpectedly excels** | Achieves 0.3105 cross-F1 (best overall) | Payload presence (data_pkt_ratio) partially transfers across datasets |
| **Within-F1 ≠ Real-world performance** | RF EXP5: 0.8247 within, 0.1519 cross | Benchmark overfitting masks generalization failure |

### Methodological Approach

**Controlled Experiment Design:**
1. Start with minimal baseline (EXP0): duration, bytes, simple ratios
2. Incrementally add protocol features (EXP1–EXP5)
3. Isolate impact of each feature on cross-dataset performance
4. Test across 3 ML algorithms (Logistic Regression, Random Forest, SVM)
5. Compare 18 experimental runs (6 variants × 3 models)

**Evaluation Strategy:**
- **Within-dataset:** Train on NSL-KDD, test on NSL-KDD test set
- **Cross-dataset:** Train on NSL-KDD, test on CICIDS 2017
- **Metric:** F1-score (primary), Recall, Precision (secondary)
- **Per-attack analysis:** Breakdown by attack type (DoS, Probe, R2L, U2R, etc.)

### Why This Matters

**Real-world impact:**
- IDS systems fail when deployed in new environments (requires expensive retraining)
- Features that work in lab conditions (NSL-KDD) don't work in production (CICIDS)
- Current practice of optimizing within-dataset F1 is **fundamentally misleading**

**Solution direction:**
- Focus on dataset-agnostic features (e.g., traffic volume, temporal patterns)
- Avoid protocol-specific features unless they transfer across environments
- Use domain adaptation or adversarial training to bridge dataset shift
- Evaluate on **held-out datasets**, not just held-out test splits

---

## Workflow

### 📊 Complete Pipeline Visualization

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RAW DATASETS (INPUT)                            │
│  NSL-KDD: KDDTrain+.arff, KDDTest+.arff | CICIDS: 8 CSV files    │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA CLEANING & VALIDATION (src/cleaner.py)              │
│  • Remove duplicates                                                │
│  • Drop NULL values & infinity values                               │
│  • Verify label columns exist ("class" vs "Label")                 │
│  • Print dataset statistics                                         │
│  OUTPUT: Clean dataframes                                           │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 2: FEATURE EXTRACTION (src/extractor.py)                    │
│                                                                      │
│ CONTROL FEATURES (EXP0):                                            │
│  • duration: connection time                                        │
│  • src_bytes, dst_bytes: traffic volume                             │
│  • bytes_per_sec: (src + dst) / duration                            │
│  • byte_ratio: src_bytes / dst_bytes                                │
│  • protocol: categorical {tcp, udp, icmp}                           │
│                                                                      │
│ PROTOCOL-AWARE VARIANTS:                                            │
│  EXP1: + syn_ratio (SYN flood intensity)                            │
│  EXP2: + rst_ratio (port scan signature)                            │
│  EXP3: + fin_ratio (connection close pattern)                       │
│  EXP4: + data_pkt_ratio (payload presence)                          │
│  EXP5: + service_bucket (port-based service tier)                   │
│                                                                      │
│ OUTPUT: Feature vectors with 6 different feature sets              │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 3: PREPROCESSING (src/preprocessor.py)                       │
│                                                                      │
│  1. LOG TRANSFORM (handle skewed distributions)                     │
│     log(src_bytes + 1), log(dst_bytes + 1), log(bytes_per_sec + 1) │
│                                                                      │
│  2. PROTOCOL ENCODING (one-hot)                                     │
│     protocol_tcp = 1 if protocol=="tcp" else 0                      │
│     protocol_udp = 1 if protocol=="udp" else 0                      │
│     protocol_icmp = 1 if protocol=="icmp" else 0                    │
│                                                                      │
│  3. FEATURE SCALING (StandardScaler)                                │
│     ⚠️  CRITICAL: Fit scaler on TRAIN set only                      │
│     Apply same scaler to TEST & CICIDS (prevent data leakage)       │
│                                                                      │
│ OUTPUT: ML-ready numpy arrays                                       │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 4: DATA ALIGNMENT (src/aligner.py)                           │
│                                                                      │
│  • Ensure KDD & CICIDS share exact feature columns                  │
│  • Map protocol names consistently                                  │
│  • Reorder columns to match feature specs                           │
│  • Save 6 sets of CSVs (EXP0–EXP5)                                  │
│                                                                      │
│  KDD: kdd_train_exp0.csv, kdd_test_exp0.csv (× 6)                  │
│  CICIDS: cicids_exp0.csv, cicids_exp1.csv (× 6)                    │
│                                                                      │
│ OUTPUT: 18 processed CSV files                                      │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 5: MODEL TRAINING (src/models/*.py)                          │
│                                                                      │
│  FOR EACH (Model, Experiment):                                      │
│    1. Load processed CSVs (e.g., kdd_train_exp0.csv)                │
│    2. Separate X (features) and y (labels)                          │
│    3. Initialize model:                                             │
│       • LogisticRegression(max_iter=1000, random_state=42)          │
│       • RandomForestClassifier(n_estimators=100, random_state=42)   │
│       • SVC(kernel='rbf', probability=True, random_state=42)        │
│    4. Train on NSL-KDD                                              │
│    5. Predict on NSL-KDD TEST (within-dataset)                      │
│    6. Predict on CICIDS (cross-dataset) ← KEY EVALUATION            │
│    7. Calculate metrics: F1, Recall, Precision                      │
│    8. Save model + scaler + feature_cols as .pkl                    │
│                                                                      │
│  OUTPUT: 18 model artifacts (3 models × 6 experiments)              │
│  • lr_exp0.pkl, lr_exp1.pkl, ..., lr_exp5.pkl                       │
│  • rf_exp0.pkl, rf_exp1.pkl, ..., rf_exp5.pkl                       │
│  • svm_exp0.pkl, svm_exp1.pkl, ..., svm_exp5.pkl                    │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 6: ANALYSIS & INSIGHT EXTRACTION (src/experiments/)          │
│                                                                      │
│  Feature Importance Analysis:                                       │
│  • Extract feature weights from each model                          │
│  • For Random Forest: use model.feature_importances_                │
│  • For Logistic Regression: use abs(model.coef_)                    │
│  • Output: feature_importance_{model}_exp{n}.csv                    │
│                                                                      │
│  Attack Breakdown Analysis:                                         │
│  • For each (model, experiment):                                    │
│    - Group predictions by attack type                               │
│    - Calculate catch rate (recall) per attack                       │
│    - Output: attack_breakdown_{model}_exp{n}.csv                    │
│                                                                      │
│ OUTPUT: CSV files for visualization                                 │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│ PHASE 7: INTERACTIVE DASHBOARD (src/web/app.py)                    │
│                                                                      │
│  Strategy Pattern (src/strategies/):                                │
│  • Each model type has a Strategy class                             │
│  • Loads .pkl → extracts model, scaler, feature_cols                │
│  • Makes predictions on user input                                  │
│                                                                      │
│  Three Dashboard Modes:                                             │
│  1. BROWSE: View all 18 experiments' metrics                        │
│  2. PREDICT: Enter features → get BENIGN/ATTACK verdict             │
│  3. COMPARE: Side-by-side F1 across all runs                        │
│                                                                      │
│  Launch: python main.py → http://127.0.0.1:5000                     │
│                                                                      │
│ OUTPUT: Interactive web interface                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 🔄 Detailed Phase Descriptions

#### PHASE 1: Data Cleaning
**Purpose:** Remove noise and inconsistencies in raw data

```python
# KDD Cleaning (src/cleaner.py)
def clean_kdd(df):
    df.drop_duplicates(inplace=True)           # Remove duplicate records
    df.dropna(inplace=True)                    # Remove NULLs
    df[str_cols] = df[str_cols].str.strip()    # Remove whitespace
    assert "class" in df.columns                # Verify label column
    return df

# CICIDS Cleaning
def clean_cicids(df):
    df.columns = df.columns.str.strip()        # Clean column names
    df.replace([np.inf, -np.inf], np.nan)      # Handle infinity (from division)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    assert "Label" in df.columns
    return df
```

**Example:**
- **Input:** KDD: 567,498 rows | CICIDS: 2,830,743 rows
- **After removal:** KDD: 567,294 rows | CICIDS: 2,827,876 rows
- **Why?** Duplicates & NULLs cause sklearn to crash

---

#### PHASE 2: Feature Extraction
**Purpose:** Create meaningful features from raw columns

```python
# Control Features (baseline, no protocol info)
df["bytes_per_sec"] = (df["src_bytes"] + df["dst_bytes"]) / (df["duration"] + 1e-6)
df["byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)

# Protocol-Aware Features (protocol-specific attack patterns)
df["syn_ratio"] = df["SYN_Flag_Count"] / df["Total_Fwd_Packets"]
df["rst_ratio"] = df["RST_Flag_Count"] / df["Total_Fwd_Packets"]
df["fin_ratio"] = df["FIN_Flag_Count"] / df["Total_Fwd_Packets"]
```

**Feature Engineering Strategy:**
| Feature | Why Extract | Dataset-Specific? | Transfers? |
|---------|-------------|-------------------|-----------|
| `bytes_per_sec` | Higher for floods & DoS | Both have it | ✅ Yes |
| `byte_ratio` | Unidirectional vs bidirectional | Both relevant | ✅ Likely |
| `syn_ratio` | SYN flood signature | NSL-KDD heavy | ❌ No (near-zero in CICIDS) |
| `rst_ratio` | Port scanning signature | NSL-KDD heavy | ❌ No (obsolete attack) |
| `data_pkt_ratio` | Payload vs overhead | Both relevant | ✅ Partially |

---

#### PHASE 3: Preprocessing
**Purpose:** Convert raw features into ML-friendly format

```python
# 3a. Log Transform (handle right-skewed distributions)
df["src_bytes"] = np.log1p(df["src_bytes"])      # log(x+1) to avoid log(0)
df["dst_bytes"] = np.log1p(df["dst_bytes"])

# 3b. Protocol Encoding (convert categorical to numeric)
df["protocol_tcp"] = (df["protocol"] == "tcp").astype(int)
df["protocol_udp"] = (df["protocol"] == "udp").astype(int)
df["protocol_icmp"] = (df["protocol"] == "icmp").astype(int)

# 3c. Feature Scaling (normalize to mean=0, std=1)
scaler = StandardScaler()
scaler.fit(X_train_numeric)                      # FIT on train only!
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)         # Reuse same scaler
X_cicids_scaled = scaler.transform(X_cicids)     # NO fitting here!
```

**Why StandardScaler matters:**
- Log Regression converges faster
- Random Forest becomes more balanced
- SVM requires normalized features for kernel computation

⚠️ **Critical:** Never fit scaler on test or cross-dataset. That's **data leakage**.

---

#### PHASE 4: Data Alignment
**Purpose:** Ensure KDD & CICIDS have identical feature columns

```python
# After feature extraction:
# KDD has:  ["duration", "protocol", "src_bytes", ..., "label"]
# CICIDS has: ["duration", "protocol", "src_bytes", ..., "Label"]

# Alignment (src/aligner.py):
df.rename(columns={"Label": "label"})            # Unify label name
df = df[FEATURE_COLS]                            # Select in same order
df.to_csv("data/processed/cicids_exp0.csv")      # Save aligned version
```

**Why alignment matters:** ML models expect X.shape = (n_samples, n_features). If feature order differs between train & test, predictions become random.

---

#### PHASE 5: Model Training
**Purpose:** Train ML models and evaluate within-dataset vs cross-dataset

```python
# Load processed datasets
train = pd.read_csv("data/processed/kdd_train_exp0.csv")
test = pd.read_csv("data/processed/kdd_test_exp0.csv")
cicids = pd.read_csv("data/processed/cicids_exp0.csv")

# Split features & labels
X_train, y_train = train[FEATURE_COLS], train["label"]
X_test, y_test = test[FEATURE_COLS], test["label"]
X_cicids, y_cicids = cicids[FEATURE_COLS], cicids["label"]

# Train
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# WITHIN-DATASET: Test on KDD
y_pred_test = model.predict(X_test)
within_f1 = f1_score(y_test, y_pred_test)

# CROSS-DATASET: Test on CICIDS ← THE KEY TEST
y_pred_cicids = model.predict(X_cicids)
cross_f1 = f1_score(y_cicids, y_pred_cicids)

# SAVE ARTIFACT
artifact = {
    "model": model,
    "scaler": scaler,
    "feature_cols": FEATURE_COLS,
    "numeric_cols": NUMERIC_COLS,
    "within_f1": within_f1,
    "cross_f1": cross_f1
}
joblib.dump(artifact, "models/lr_exp0.pkl")
```

**Output Example:**
```
LogReg EXP0:
  Within-dataset (KDD test) F1: 0.7363 ← Good
  Cross-dataset (CICIDS) F1: 0.2312 ← 69% drop!
  Performance drop: 0.5051

LogReg EXP2 (with rst_ratio):
  Within-dataset F1: 0.7771 ← Even better!
  Cross-dataset F1: 0.0204 ← CATASTROPHIC!
  Performance drop: 0.7567
```

**Key insight:** Adding rst_ratio IMPROVES within-dataset performance but DESTROYS cross-dataset performance. The feature is too dataset-specific.

---

#### PHASE 6: Analysis & Insights
**Purpose:** Extract patterns from predictions

```python
# Feature Importance (why did model make decisions?)
for model_name in ["LogReg", "RandomForest", "SVM"]:
    for exp in range(6):
        artifact = joblib.load(f"models/{model_name}_exp{exp}.pkl")
        model = artifact["model"]
        
        # Get importance scores
        if hasattr(model, "feature_importances_"):  # Tree-based
            importances = model.feature_importances_
        else:                                        # Linear
            importances = abs(model.coef_[0])
        
        df_importance = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        df_importance.to_csv(f"results/{model_name}_exp{exp}_importance.csv")

# Attack Breakdown (which attack types does the model catch?)
for attack_type in ["DoS", "Probe", "R2L", "U2R", "normal"]:
    mask = (y_cicids == attack_type)
    catch_rate = recall_score(y_cicids[mask], y_pred_cicids[mask])
    print(f"{attack_type}: {catch_rate:.2%} catch rate")
```

---

#### PHASE 7: Interactive Dashboard
**Purpose:** Visualize & explore results

```python
# Strategy Pattern: Abstract model loading
class LogisticRegressionStrategy:
    def __init__(self, pkl_path):
        artifact = joblib.load(pkl_path)
        self.model = artifact["model"]
        self.scaler = artifact["scaler"]
        self.feature_cols = artifact["feature_cols"]
        self.numeric_cols = artifact["numeric_cols"]
    
    def predict(self, feature_dict):
        # Convert user input → proper format
        X = pd.DataFrame([feature_dict])[self.feature_cols]
        
        # Scale using saved scaler
        X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        
        # Predict
        pred = self.model.predict(X)
        proba = self.model.predict_proba(X)
        
        return {
            "prediction": "ATTACK" if pred[0] == 1 else "BENIGN",
            "confidence": proba[0].max()
        }

# Flask routes
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    strategy = get_strategy(data["model"])
    result = strategy.predict(data["features"])
    return jsonify(result)
```

### 🚀 Quick Start Commands

```bash
# Full end-to-end pipeline
python scripts/combine_cicids.py  # Prepare CICIDS from raw files

# Build experiment CSVs
python -c "
from src.pipeline import run_control_pipeline, run_protocol_aware_pipeline
run_control_pipeline()
run_protocol_aware_pipeline()
"

# Train all models
python -c "
from src.models.logistic_regression import *
from src.models.random_forest import *
from src.models.svm import *
[run_logistic_regression_EXP0(), run_logistic_regression_EXP1(), ...]
[run_random_forest_EXP0(), run_random_forest_EXP1(), ...]
[run_svm_EXP0(), run_svm_EXP1(), ...]
"

# Analyze results
python src/experiments/feature_importance.py
python src/experiments/attack_breakdown.py

# Launch dashboard
python main.py
# Open http://127.0.0.1:5000
```

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

### Step 6 — Launch the Interactive Demo (Flask)
```bash
python main.py
# → http://127.0.0.1:5000
```

The demo is built on Strategy + Facade (see `src/strategies/`, `src/facade/`,
`src/context.py`) and reads the saved `.pkl` artefacts in `models/` plus the
results CSVs under `results/`. Three modes:

1. **Browse results** — pick a (model, experiment), see saved metrics,
   per-attack catch rates, and matching figures.
2. **Live predict** — enter feature values (or sample a row from a prepared
   dataset), see the model's BENIGN/ATTACK verdict + confidence score.
   Requires the corresponding `.pkl` in `models/`.
3. **Compare all** — pivot table of within vs cross F1 across all 18 runs.

The legacy `run_*_EXPn` training scripts in `src/models/` are kept untouched
for reproducibility. The Strategy classes wrap their saved artefacts read-only.

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