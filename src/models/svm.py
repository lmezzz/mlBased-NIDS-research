import joblib
import pandas as pd
import os
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, f1_score, recall_score, precision_score
)
from src.preprocessor import fit_scaler, apply_scaler
from config import (
    KDD_TRAIN_CONTROL, KDD_TEST_CONTROL, CICIDS_CONTROL,
    KDD_TRAIN_EXP1, KDD_TEST_EXP1, CICIDS_EXP1,
    KDD_TRAIN_EXP2, KDD_TEST_EXP2, CICIDS_EXP2,
    KDD_TRAIN_EXP3, KDD_TEST_EXP3, CICIDS_EXP3,
    KDD_TRAIN_EXP4, KDD_TEST_EXP4, CICIDS_EXP4,
    KDD_TRAIN_EXP5, KDD_TEST_EXP5, CICIDS_EXP5, MODELS_DIR, RESULTS_DIR,
)

NUMERIC_COLS = ["duration", "bytes_per_sec", "byte_ratio"]


def _run(train_path, test_path, cicids_path, feature_cols, exp_name):
    print(f"\n[SVM] Loading {exp_name}...")
    train  = pd.read_csv(train_path)
    test   = pd.read_csv(test_path)
    cicids = pd.read_csv(cicids_path)

    # ── SCALE ──
    scaler, _ = fit_scaler(train[NUMERIC_COLS])
    train[NUMERIC_COLS]  = apply_scaler(train,  scaler, NUMERIC_COLS)[NUMERIC_COLS]
    test[NUMERIC_COLS]   = apply_scaler(test,   scaler, NUMERIC_COLS)[NUMERIC_COLS]
    cicids[NUMERIC_COLS] = apply_scaler(cicids, scaler, NUMERIC_COLS)[NUMERIC_COLS]

    # ── SPLIT ──
    X_train  = train[feature_cols]
    y_train  = train["label"]
    X_test   = test[feature_cols]
    y_test   = test["label"]
    X_cicids = cicids[feature_cols]
    y_cicids = cicids["label"]

    # ── TRAIN ──
    # LinearSVC — much faster than SVC on large datasets
    # max_iter=2000 to ensure convergence
    print(f"[SVM] Training {exp_name}...")
    model = LinearSVC(max_iter=2000, random_state=42)
    model.fit(X_train, y_train)

    # ── PHASE 1 ──
    print(f"\n[SVM] {exp_name} — Phase 1 (within-dataset):")
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    print(f"F1:        {f1_score(y_test, y_pred_test):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_test):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")

    # ── PHASE 2 ──
    print(f"\n[SVM] {exp_name} — Phase 2 (cross-dataset):")
    y_pred_cicids = model.predict(X_cicids)
    print(classification_report(y_cicids, y_pred_cicids))
    print(f"F1:        {f1_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Recall:    {recall_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Precision: {precision_score(y_cicids, y_pred_cicids):.4f}")

    # ── SUMMARY ──
    f1_within = f1_score(y_test, y_pred_test)
    f1_cross  = f1_score(y_cicids, y_pred_cicids)
    print(f"\n[SVM] ── SUMMARY {exp_name} ──")
    print(f"Within-dataset F1:  {f1_within:.4f}")
    print(f"Cross-dataset F1:   {f1_cross:.4f}")
    print(f"Performance drop:   {f1_within - f1_cross:.4f}")
    results = {
        "model": "SVM",
        "experiment": exp_name,
        "within_f1": f1_within,
        "cross_f1": f1_cross,
        "within_recall": recall_score(y_test, y_pred_test),
        "cross_recall": recall_score(y_cicids, y_pred_cicids),
        "within_precision": precision_score(y_test, y_pred_test),
        "cross_precision": precision_score(y_cicids, y_pred_cicids),
        "performance_drop": f1_within - f1_cross,
        "num_features": len(feature_cols)
    }

    results_file = RESULTS_DIR / "all_results.csv"

    if results_file.exists():
        df = pd.read_csv(results_file)
        df = df[~((df["model"] == "SVM") &
                (df["experiment"] == exp_name))]
        df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)
    else:
        df = pd.DataFrame([results])

    df.to_csv(results_file, index=False)
    print(f"[SVM] Results saved → {results_file}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "features": feature_cols , "numeric_cols": NUMERIC_COLS}, 
                MODELS_DIR / f"svm_{exp_name.lower()}.pkl")
    print(f"[SVM] Model saved → svm_{exp_name.lower()}.pkl")

    return model, scaler


# ── EXP 0 ──
def run_svm_EXP0():
    FEATURE_COLS = [
        "duration", "bytes_per_sec", "byte_ratio",
        "protocol_tcp", "protocol_udp", "protocol_icmp",
        "src_bytes", "dst_bytes"
    ]
    return _run(KDD_TRAIN_CONTROL, KDD_TEST_CONTROL, CICIDS_CONTROL,
                FEATURE_COLS, "EXP0")

# ── EXP 1 ──
def run_svm_EXP1():
    FEATURE_COLS = [
        "duration", "bytes_per_sec", "byte_ratio",
        "protocol_tcp", "protocol_udp", "protocol_icmp",
        "syn_ratio", "dst_bytes"
    ]
    return _run(KDD_TRAIN_EXP1, KDD_TEST_EXP1, CICIDS_EXP1,
                FEATURE_COLS, "EXP1")

# ── EXP 2 ──
def run_svm_EXP2():
    FEATURE_COLS = [
        "duration", "bytes_per_sec", "byte_ratio",
        "protocol_tcp", "protocol_udp", "protocol_icmp",
        "syn_ratio", "rst_ratio"
    ]
    return _run(KDD_TRAIN_EXP2, KDD_TEST_EXP2, CICIDS_EXP2,
                FEATURE_COLS, "EXP2")

# ── EXP 3 ──
def run_svm_EXP3():
    FEATURE_COLS = [
        "duration", "bytes_per_sec", "byte_ratio",
        "protocol_tcp", "protocol_udp", "protocol_icmp",
        "syn_ratio", "rst_ratio", "fin_ratio"
    ]
    return _run(KDD_TRAIN_EXP3, KDD_TEST_EXP3, CICIDS_EXP3,
                FEATURE_COLS, "EXP3")

# ── EXP 4 ──
def run_svm_EXP4():
    FEATURE_COLS = [
        "duration", "bytes_per_sec", "byte_ratio",
        "protocol_tcp", "protocol_udp", "protocol_icmp",
        "syn_ratio", "rst_ratio", "fin_ratio", "data_pkt_ratio"
    ]
    return _run(KDD_TRAIN_EXP4, KDD_TEST_EXP4, CICIDS_EXP4,
                FEATURE_COLS, "EXP4")

# ── EXP 5 ──
def run_svm_EXP5():
    FEATURE_COLS = [
        "duration", "bytes_per_sec", "byte_ratio",
        "protocol_tcp", "protocol_udp", "protocol_icmp",
        "syn_ratio", "rst_ratio", "fin_ratio", "data_pkt_ratio",
        "svc_http", "svc_ftp", "svc_dns", "svc_ssh",
        "svc_https", "svc_smtp", "svc_telnet", "svc_mysql",
        "svc_rdp", "svc_system", "svc_user", "svc_dynamic", "svc_unknown"
    ]
    return _run(KDD_TRAIN_EXP5, KDD_TEST_EXP5, CICIDS_EXP5,
                FEATURE_COLS, "EXP5")