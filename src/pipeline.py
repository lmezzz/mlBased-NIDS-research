import pandas as pd
from src.loader import load_KDD_TRAIN, load_KDD_TEST, load_CICI_COMBINED
from src.cleaner import clean_kdd, clean_cicids
from src.extractor import feature_control_kdd, feature_control_cicids
from src.aligner import align_kdd, align_cicids
from src.preprocessor import encode_protocol, log_transform, fit_scaler, apply_scaler
from config import KDD_TRAIN_CONTROL, KDD_TEST_CONTROL, CICIDS_CONTROL

def run_control_pipeline():
    print("[Pipeline] Running control feature pipeline...")

    # ── KDD TRAIN ──
    train = load_KDD_TRAIN()
    train = clean_kdd(train)
    train = feature_control_kdd(train)
    train = log_transform(train)
    train = encode_protocol(train)
    train = align_kdd(train)

    # ── KDD TEST ──
    test = load_KDD_TEST()
    test = clean_kdd(test)
    test = feature_control_kdd(test)
    test = log_transform(test)
    test = encode_protocol(test)
    test = align_kdd(test)

    # ── CICIDS ──
    cicids = load_CICI_COMBINED()
    cicids = clean_cicids(cicids)
    cicids = feature_control_cicids(cicids)
    cicids = log_transform(cicids)
    cicids["protocol_icmp"] = False # Ensure ICMP column exists for CICIDS
    cicids = encode_protocol(cicids)
    cicids = align_cicids(cicids)

    # ── SAVE ──
    train.to_csv(KDD_TRAIN_CONTROL, index=False)
    test.to_csv(KDD_TEST_CONTROL, index=False)
    cicids.to_csv(CICIDS_CONTROL, index=False)

    print(f"[Pipeline] KDD train  → {train.shape}")
    print(f"[Pipeline] KDD train  → {train.head(2)}")
    print(f"[Pipeline] KDD test   → {test.shape}")
    print(f"[Pipeline] KDD test   → {test.head(2)}")
    print(f"[Pipeline] CICIDS     → {cicids.shape}")
    print(f"[Pipeline] CICIDS     → {cicids.head(2)}")
    print("[Pipeline] Done ✓")