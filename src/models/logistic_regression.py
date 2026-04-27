import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
from src.preprocessor import fit_scaler, apply_scaler
from config import CICIDS_EXP1, CICIDS_EXP2, KDD_TRAIN_CONTROL, KDD_TEST_CONTROL, CICIDS_CONTROL , KDD_TRAIN_EXP1, KDD_TEST_EXP1 , KDD_TRAIN_EXP2, KDD_TEST_EXP2,KDD_TRAIN_EXP3, KDD_TEST_EXP3, CICIDS_EXP3 , KDD_TRAIN_EXP4, KDD_TEST_EXP4, CICIDS_EXP4 , KDD_TRAIN_EXP5, KDD_TEST_EXP5, CICIDS_EXP5



def run_logistic_regression():

    NUMERIC_COLS = ["duration", "src_bytes", "dst_bytes", "bytes_per_sec", "byte_ratio"]
    FEATURE_COLS = ["duration", "src_bytes", "dst_bytes", "bytes_per_sec", "byte_ratio",
                "protocol_tcp", "protocol_udp", "protocol_icmp"]

    print("[LogReg] Loading processed data...")
    train  = pd.read_csv(KDD_TRAIN_CONTROL)
    test   = pd.read_csv(KDD_TEST_CONTROL)
    cicids = pd.read_csv(CICIDS_CONTROL)

    # ── SCALE ──
    scaler , _ = fit_scaler(train[NUMERIC_COLS])
    train[NUMERIC_COLS]  = apply_scaler(train,  scaler, NUMERIC_COLS)[NUMERIC_COLS]
    test[NUMERIC_COLS]   = apply_scaler(test,   scaler, NUMERIC_COLS)[NUMERIC_COLS]
    cicids[NUMERIC_COLS] = apply_scaler(cicids, scaler, NUMERIC_COLS)[NUMERIC_COLS]

    # ── SPLIT ──
    X_train  = train[FEATURE_COLS]
    y_train  = train["label"]
    X_test   = test[FEATURE_COLS]
    y_test   = test["label"]
    X_cicids = cicids[FEATURE_COLS]
    y_cicids = cicids["label"]

    # ── TRAIN ──
    print("[LogReg] Training...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # ── PHASE 1 — WITHIN DATASET ──
    print("\n[LogReg] Phase 1 — KDD test (within-dataset):")
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    print(f"F1:        {f1_score(y_test, y_pred_test):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_test):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")

    # ── PHASE 2 — CROSS DATASET ──
    print("\n[LogReg] Phase 2 — CICIDS (cross-dataset):")
    y_pred_cicids = model.predict(X_cicids)
    print(classification_report(y_cicids, y_pred_cicids))
    print(f"F1:        {f1_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Recall:    {recall_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Precision: {precision_score(y_cicids, y_pred_cicids):.4f}")

    # ── SUMMARY ──
    f1_within = f1_score(y_test, y_pred_test)
    f1_cross  = f1_score(y_cicids, y_pred_cicids)
    print(f"\n[LogReg] ── SUMMARY ──")
    print(f"Within-dataset F1:  {f1_within:.4f}")
    print(f"Cross-dataset F1:   {f1_cross:.4f}")
    print(f"Performance drop:   {f1_within - f1_cross:.4f}")

    return model, scaler


def run_logistic_regression_EXP1():

    NUMERIC_COLS = ["duration", "dst_bytes", "bytes_per_sec", "byte_ratio"]
    FEATURE_COLS = ["duration", "dst_bytes", "syn_ratio", "bytes_per_sec", "byte_ratio",
                "protocol_tcp", "protocol_udp", "protocol_icmp"]


    print("[LogReg] Loading processed data...")
    train  = pd.read_csv(KDD_TRAIN_EXP1)
    test   = pd.read_csv(KDD_TEST_EXP1)
    cicids = pd.read_csv(CICIDS_EXP1)

    # ── SCALE ──
    scaler , _ = fit_scaler(train[NUMERIC_COLS])
    train[NUMERIC_COLS]  = apply_scaler(train,  scaler, NUMERIC_COLS)[NUMERIC_COLS]
    test[NUMERIC_COLS]   = apply_scaler(test,   scaler, NUMERIC_COLS)[NUMERIC_COLS]
    cicids[NUMERIC_COLS] = apply_scaler(cicids, scaler, NUMERIC_COLS)[NUMERIC_COLS]

    # ── SPLIT ──
    X_train  = train[FEATURE_COLS]
    y_train  = train["label"]
    X_test   = test[FEATURE_COLS]
    y_test   = test["label"]
    X_cicids = cicids[FEATURE_COLS]
    y_cicids = cicids["label"]

    # ── TRAIN ──
    print("[LogReg] Training...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # ── PHASE 1 — WITHIN DATASET ──
    print("\n[LogReg] Phase 1 — KDD test (within-dataset):")
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    print(f"F1:        {f1_score(y_test, y_pred_test):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_test):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")

    # ── PHASE 2 — CROSS DATASET ──
    print("\n[LogReg] Phase 2 — CICIDS (cross-dataset):")
    y_pred_cicids = model.predict(X_cicids)
    print(classification_report(y_cicids, y_pred_cicids))
    print(f"F1:        {f1_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Recall:    {recall_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Precision: {precision_score(y_cicids, y_pred_cicids):.4f}")

    # ── SUMMARY ──
    f1_within = f1_score(y_test, y_pred_test)
    f1_cross  = f1_score(y_cicids, y_pred_cicids)
    print(f"\n[LogReg] ── SUMMARY ──")
    print(f"Within-dataset F1:  {f1_within:.4f}")
    print(f"Cross-dataset F1:   {f1_cross:.4f}")
    print(f"Performance drop:   {f1_within - f1_cross:.4f}")

    return model, scaler

def run_logistic_regression_EXP2():

    NUMERIC_COLS = ["duration", "bytes_per_sec", "byte_ratio"]
    FEATURE_COLS = ["duration", "rst_ratio", "syn_ratio", "bytes_per_sec", "byte_ratio",
                "protocol_tcp", "protocol_udp", "protocol_icmp"]


    print("[LogReg] Loading processed data...")
    train  = pd.read_csv(KDD_TRAIN_EXP2)
    test   = pd.read_csv(KDD_TEST_EXP2)
    cicids = pd.read_csv(CICIDS_EXP2)

    # ── SCALE ──
    scaler , _ = fit_scaler(train[NUMERIC_COLS])
    train[NUMERIC_COLS]  = apply_scaler(train,  scaler, NUMERIC_COLS)[NUMERIC_COLS]
    test[NUMERIC_COLS]   = apply_scaler(test,   scaler, NUMERIC_COLS)[NUMERIC_COLS]
    cicids[NUMERIC_COLS] = apply_scaler(cicids, scaler, NUMERIC_COLS)[NUMERIC_COLS]

    # ── SPLIT ──
    X_train  = train[FEATURE_COLS]
    y_train  = train["label"]
    X_test   = test[FEATURE_COLS]
    y_test   = test["label"]
    X_cicids = cicids[FEATURE_COLS]
    y_cicids = cicids["label"]

    # ── TRAIN ──
    print("[LogReg] Training...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # ── PHASE 1 — WITHIN DATASET ──
    print("\n[LogReg] Phase 1 — KDD test (within-dataset):")
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    print(f"F1:        {f1_score(y_test, y_pred_test):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_test):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")

    # ── PHASE 2 — CROSS DATASET ──
    print("\n[LogReg] Phase 2 — CICIDS (cross-dataset):")
    y_pred_cicids = model.predict(X_cicids)
    print(classification_report(y_cicids, y_pred_cicids))
    print(f"F1:        {f1_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Recall:    {recall_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Precision: {precision_score(y_cicids, y_pred_cicids):.4f}")

    # ── SUMMARY ──
    f1_within = f1_score(y_test, y_pred_test)
    f1_cross  = f1_score(y_cicids, y_pred_cicids)
    print(f"\n[LogReg] ── SUMMARY ──")
    print(f"Within-dataset F1:  {f1_within:.4f}")
    print(f"Cross-dataset F1:   {f1_cross:.4f}")
    print(f"Performance drop:   {f1_within - f1_cross:.4f}")

    return model, scaler

def run_logistic_regression_EXP3():

    NUMERIC_COLS = ["duration", "bytes_per_sec", "byte_ratio"]
    FEATURE_COLS = ["duration", "fin_ratio","rst_ratio", "syn_ratio", "bytes_per_sec", "byte_ratio",
                "protocol_tcp", "protocol_udp", "protocol_icmp"]


    print("[LogReg] Loading processed data...")
    train  = pd.read_csv(KDD_TRAIN_EXP3)
    test   = pd.read_csv(KDD_TEST_EXP3)
    cicids = pd.read_csv(CICIDS_EXP3)

    # ── SCALE ──
    scaler , _ = fit_scaler(train[NUMERIC_COLS])
    train[NUMERIC_COLS]  = apply_scaler(train,  scaler, NUMERIC_COLS)[NUMERIC_COLS]
    test[NUMERIC_COLS]   = apply_scaler(test,   scaler, NUMERIC_COLS)[NUMERIC_COLS]
    cicids[NUMERIC_COLS] = apply_scaler(cicids, scaler, NUMERIC_COLS)[NUMERIC_COLS]

    # ── SPLIT ──
    X_train  = train[FEATURE_COLS]
    y_train  = train["label"]
    X_test   = test[FEATURE_COLS]
    y_test   = test["label"]
    X_cicids = cicids[FEATURE_COLS]
    y_cicids = cicids["label"]

    # ── TRAIN ──
    print("[LogReg] Training...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # ── PHASE 1 — WITHIN DATASET ──
    print("\n[LogReg] Phase 1 — KDD test (within-dataset):")
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    print(f"F1:        {f1_score(y_test, y_pred_test):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_test):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")

    # ── PHASE 2 — CROSS DATASET ──
    print("\n[LogReg] Phase 2 — CICIDS (cross-dataset):")
    y_pred_cicids = model.predict(X_cicids)
    print(classification_report(y_cicids, y_pred_cicids))
    print(f"F1:        {f1_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Recall:    {recall_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Precision: {precision_score(y_cicids, y_pred_cicids):.4f}")

    # ── SUMMARY ──
    f1_within = f1_score(y_test, y_pred_test)
    f1_cross  = f1_score(y_cicids, y_pred_cicids)
    print(f"\n[LogReg] ── SUMMARY ──")
    print(f"Within-dataset F1:  {f1_within:.4f}")
    print(f"Cross-dataset F1:   {f1_cross:.4f}")
    print(f"Performance drop:   {f1_within - f1_cross:.4f}")

    return model, scaler

def run_logistic_regression_EXP4():
    NUMERIC_COLS = ["duration", "bytes_per_sec", "byte_ratio"]
    FEATURE_COLS = ["duration", "data_pkt_ratio", "bytes_per_sec", "byte_ratio",
                "protocol_tcp", "protocol_udp", "protocol_icmp"]


    print("[LogReg] Loading processed data...")
    train  = pd.read_csv(KDD_TRAIN_EXP4)
    test   = pd.read_csv(KDD_TEST_EXP4)
    cicids = pd.read_csv(CICIDS_EXP4)

    # ── SCALE ──
    scaler , _ = fit_scaler(train[NUMERIC_COLS])
    train[NUMERIC_COLS]  = apply_scaler(train,  scaler, NUMERIC_COLS)[NUMERIC_COLS]
    test[NUMERIC_COLS]   = apply_scaler(test,   scaler, NUMERIC_COLS)[NUMERIC_COLS]
    cicids[NUMERIC_COLS] = apply_scaler(cicids, scaler, NUMERIC_COLS)[NUMERIC_COLS]

    # ── SPLIT ──
    X_train  = train[FEATURE_COLS]
    y_train  = train["label"]
    X_test   = test[FEATURE_COLS]
    y_test   = test["label"]
    X_cicids = cicids[FEATURE_COLS]
    y_cicids = cicids["label"]

    # ── TRAIN ──
    print("[LogReg] Training...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # ── PHASE 1 — WITHIN DATASET ──
    print("\n[LogReg] Phase 1 — KDD test (within-dataset):")
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    print(f"F1:        {f1_score(y_test, y_pred_test):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_test):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")

    # ── PHASE 2 — CROSS DATASET ──
    print("\n[LogReg] Phase 2 — CICIDS (cross-dataset):")
    y_pred_cicids = model.predict(X_cicids)
    print(classification_report(y_cicids, y_pred_cicids))
    print(f"F1:        {f1_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Recall:    {recall_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Precision: {precision_score(y_cicids, y_pred_cicids):.4f}")

    # ── SUMMARY ──
    f1_within = f1_score(y_test, y_pred_test)
    f1_cross  = f1_score(y_cicids, y_pred_cicids)
    print(f"\n[LogReg] ── SUMMARY ──")
    print(f"Within-dataset F1:  {f1_within:.4f}")
    print(f"Cross-dataset F1:   {f1_cross:.4f}")
    print(f"Performance drop:   {f1_within - f1_cross:.4f}")

    return model, scaler

def run_logistic_regression_EXP5():
    NUMERIC_COLS = ["duration", "bytes_per_sec", "byte_ratio"]
    FEATURE_COLS = ["data_pkt_ratio","fin_ratio", "syn_ratio", "bytes_per_sec", "byte_ratio",
                "protocol_tcp", "protocol_udp", "protocol_icmp" , "svc_http", "svc_ftp", "svc_dns" , "svc_ssh", 
                "svc_https" , "svc_smtp" , "svc_telnet" , "svc_mysql" , "svc_rdp" , "svc_system" , "svc_user" , "svc_dynamic" , "svc_unknown"]


    print("[LogReg] Loading processed data...")
    train  = pd.read_csv(KDD_TRAIN_EXP5)
    test   = pd.read_csv(KDD_TEST_EXP5)
    cicids = pd.read_csv(CICIDS_EXP5)

    # ── SCALE ──
    scaler , _ = fit_scaler(train[NUMERIC_COLS])
    train[NUMERIC_COLS]  = apply_scaler(train,  scaler, NUMERIC_COLS)[NUMERIC_COLS]
    test[NUMERIC_COLS]   = apply_scaler(test,   scaler, NUMERIC_COLS)[NUMERIC_COLS]
    cicids[NUMERIC_COLS] = apply_scaler(cicids, scaler, NUMERIC_COLS)[NUMERIC_COLS]

    # ── SPLIT ──
    X_train  = train[FEATURE_COLS]
    y_train  = train["label"]
    X_test   = test[FEATURE_COLS]
    y_test   = test["label"]
    X_cicids = cicids[FEATURE_COLS]
    y_cicids = cicids["label"]

    # ── TRAIN ──
    print("[LogReg] Training...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # ── PHASE 1 — WITHIN DATASET ──
    print("\n[LogReg] Phase 1 — KDD test (within-dataset):")
    y_pred_test = model.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    print(f"F1:        {f1_score(y_test, y_pred_test):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred_test):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_test):.4f}")

    # ── PHASE 2 — CROSS DATASET ──
    print("\n[LogReg] Phase 2 — CICIDS (cross-dataset):")
    y_pred_cicids = model.predict(X_cicids)
    print(classification_report(y_cicids, y_pred_cicids))
    print(f"F1:        {f1_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Recall:    {recall_score(y_cicids, y_pred_cicids):.4f}")
    print(f"Precision: {precision_score(y_cicids, y_pred_cicids):.4f}")

    # ── SUMMARY ──
    f1_within = f1_score(y_test, y_pred_test)
    f1_cross  = f1_score(y_cicids, y_pred_cicids)
    print(f"\n[LogReg] ── SUMMARY ──")
    print(f"Within-dataset F1:  {f1_within:.4f}")
    print(f"Cross-dataset F1:   {f1_cross:.4f}")
    print(f"Performance drop:   {f1_within - f1_cross:.4f}")

    return model, scaler


def run_logistic_regression_all_experiments():
    print("\n\n=== LOGISTIC REGRESSION CONTROL ===")
    run_logistic_regression()
    print("\n\n=== LOGISTIC REGRESSION EXPERIMENT 1 ===")
    run_logistic_regression_EXP1()
    print("\n\n=== LOGISTIC REGRESSION EXPERIMENT 2 ===")
    run_logistic_regression_EXP2()
    print("\n\n=== LOGISTIC REGRESSION EXPERIMENT 3 ===")
    run_logistic_regression_EXP3()
    print("\n\n=== LOGISTIC REGRESSION EXPERIMENT 4 ===")
    run_logistic_regression_EXP4()
    print("\n\n=== LOGISTIC REGRESSION EXPERIMENT 5 ===")
    run_logistic_regression_EXP5()