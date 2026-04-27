import pandas as pd
from src.loader import load_KDD_TRAIN, load_KDD_TEST, load_CICI_COMBINED , load_KDD_CONTROL, load_KDD_TEST_CONTROL , load_CICI_CONTROL
from src.cleaner import clean_kdd, clean_cicids
from src.extractor import feature_control_kdd, feature_control_cicids , feature_protocol_aware_kdd, feature_protocol_aware_cicids
from src.aligner import align_kdd, align_cicids , align_protocol_aware_kdd, align_protocol_aware_cicids
from src.preprocessor import encode_protocol, log_transform, fit_scaler, apply_scaler
from config import KDD_TRAIN_CONTROL, KDD_TEST_CONTROL, CICIDS_CONTROL, PROTOCOL_AWARE_CICIDS, PROTOCOL_AWARE_KDD_TEST, PROTOCOL_AWARE_KDD_TRAIN

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

def run_protocol_aware_pipeline():
    print("[Pipeline] Running protocol-aware feature pipeline...")
    train_aware_kdd = load_KDD_TRAIN()
    train_aware_kdd = clean_kdd(train_aware_kdd)
    train_aware_kdd = feature_protocol_aware_kdd(train_aware_kdd)
    train_aware_kdd = log_transform(train_aware_kdd)
    train_aware_kdd = align_protocol_aware_kdd(train_aware_kdd)

    test_aware_kdd = load_KDD_TEST()
    test_aware_kdd = clean_kdd(test_aware_kdd)
    test_aware_kdd = feature_protocol_aware_kdd(test_aware_kdd)
    test_aware_kdd = log_transform(test_aware_kdd)
    test_aware_kdd = align_protocol_aware_kdd(test_aware_kdd)

    aware_cicids = load_CICI_COMBINED()
    aware_cicids = clean_cicids(aware_cicids)
    aware_cicids = feature_protocol_aware_cicids(aware_cicids)
    aware_cicids = log_transform(aware_cicids)
    aware_cicids = align_protocol_aware_cicids(aware_cicids)

    train_aware_kdd.to_csv(PROTOCOL_AWARE_KDD_TRAIN, index=False)
    test_aware_kdd.to_csv(PROTOCOL_AWARE_KDD_TEST, index=False)
    aware_cicids.to_csv(PROTOCOL_AWARE_CICIDS, index=False)
    print("[Pipeline] Protocol-aware pipeline implemented.")    


from config import (
    KDD_TRAIN_CONTROL, KDD_TEST_CONTROL, CICIDS_CONTROL,
    PROTOCOL_AWARE_KDD_TRAIN, PROTOCOL_AWARE_KDD_TEST, PROTOCOL_AWARE_CICIDS,
    KDD_TRAIN_EXP1, KDD_TEST_EXP1, CICIDS_EXP1,
    KDD_TRAIN_EXP2, KDD_TEST_EXP2, CICIDS_EXP2,
    KDD_TRAIN_EXP3, KDD_TEST_EXP3, CICIDS_EXP3,
    KDD_TRAIN_EXP4, KDD_TEST_EXP4, CICIDS_EXP4,
    KDD_TRAIN_EXP5, KDD_TEST_EXP5, CICIDS_EXP5,
)
from src.preprocessor import encode_service_bucket


# ── HELPERS ──

def _load_base():
    """Load all 6 existing processed CSVs once."""
    ctrl_train  = pd.read_csv(KDD_TRAIN_CONTROL)
    ctrl_test   = pd.read_csv(KDD_TEST_CONTROL)
    ctrl_cicids = pd.read_csv(CICIDS_CONTROL)
    prot_train  = pd.read_csv(PROTOCOL_AWARE_KDD_TRAIN)
    prot_test   = pd.read_csv(PROTOCOL_AWARE_KDD_TEST)
    prot_cicids = pd.read_csv(PROTOCOL_AWARE_CICIDS)
    print("[Pipeline] Base CSVs loaded ✓")
    return ctrl_train, ctrl_test, ctrl_cicids, prot_train, prot_test, prot_cicids


def _save(train, test, cicids, paths, name):
    t, te, c = paths
    train.to_csv(t,  index=False)
    test.to_csv(te,  index=False)
    cicids.to_csv(c, index=False)
    print(f"[Pipeline] {name} → train{train.shape} test{test.shape} cicids{cicids.shape} ✓")


def _combine(ctrl, prot, cols):
    """
    Merge control and protocol-aware dataframes,
    select only the columns needed for this experiment.
    label column always carried from ctrl.
    """
    merged = pd.concat([ctrl, prot], axis=1)
    # avoid duplicate label columns if both have it
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged[cols + ["label"]]


# ── CONTROL COLS (base — always present) ──
CTRL_BASE = [
    "duration", "bytes_per_sec", "byte_ratio",
    "protocol_tcp", "protocol_udp", "protocol_icmp",
]


# ════════════════════════════════════════
# EXP 1 — swap src_bytes → syn_ratio
# ════════════════════════════════════════
def run_exp1_pipeline():
    ctrl_train, ctrl_test, ctrl_cicids, \
    prot_train, prot_test, prot_cicids = _load_base()

    COLS = CTRL_BASE + ["syn_ratio", "dst_bytes"]

    train  = _combine(ctrl_train,  prot_train,  COLS)
    test   = _combine(ctrl_test,   prot_test,   COLS)
    cicids = _combine(ctrl_cicids, prot_cicids, COLS)

    _save(train, test, cicids,
          (KDD_TRAIN_EXP1, KDD_TEST_EXP1, CICIDS_EXP1),
          "Exp1 (syn_ratio replaces src_bytes)")


# ════════════════════════════════════════
# EXP 2 — swap dst_bytes → rst_ratio
# ════════════════════════════════════════
def run_exp2_pipeline():
    ctrl_train, ctrl_test, ctrl_cicids, \
    prot_train, prot_test, prot_cicids = _load_base()

    COLS = CTRL_BASE + ["syn_ratio", "rst_ratio"]

    train  = _combine(ctrl_train,  prot_train,  COLS)
    test   = _combine(ctrl_test,   prot_test,   COLS)
    cicids = _combine(ctrl_cicids, prot_cicids, COLS)

    _save(train, test, cicids,
          (KDD_TRAIN_EXP2, KDD_TEST_EXP2, CICIDS_EXP2),
          "Exp2 (rst_ratio replaces dst_bytes)")


# ════════════════════════════════════════
# EXP 3 — add fin_ratio
# ════════════════════════════════════════
def run_exp3_pipeline():
    ctrl_train, ctrl_test, ctrl_cicids, \
    prot_train, prot_test, prot_cicids = _load_base()

    COLS = CTRL_BASE + ["syn_ratio", "rst_ratio", "fin_ratio"]

    train  = _combine(ctrl_train,  prot_train,  COLS)
    test   = _combine(ctrl_test,   prot_test,   COLS)
    cicids = _combine(ctrl_cicids, prot_cicids, COLS)

    _save(train, test, cicids,
          (KDD_TRAIN_EXP3, KDD_TEST_EXP3, CICIDS_EXP3),
          "Exp3 (+ fin_ratio)")


# ════════════════════════════════════════
# EXP 4 — add data_pkt_ratio
# ════════════════════════════════════════
def run_exp4_pipeline():
    ctrl_train, ctrl_test, ctrl_cicids, \
    prot_train, prot_test, prot_cicids = _load_base()

    COLS = CTRL_BASE + ["syn_ratio", "rst_ratio", "fin_ratio", "data_pkt_ratio"]

    train  = _combine(ctrl_train,  prot_train,  COLS)
    test   = _combine(ctrl_test,   prot_test,   COLS)
    cicids = _combine(ctrl_cicids, prot_cicids, COLS)

    _save(train, test, cicids,
          (KDD_TRAIN_EXP4, KDD_TEST_EXP4, CICIDS_EXP4),
          "Exp4 (+ data_pkt_ratio)")


# ════════════════════════════════════════
# EXP 5 — add service_bucket (one-hot)
# ════════════════════════════════════════
def run_exp5_pipeline():
    ctrl_train, ctrl_test, ctrl_cicids, \
    prot_train, prot_test, prot_cicids = _load_base()

    COLS = CTRL_BASE + [
        "syn_ratio", "rst_ratio", "fin_ratio",
        "data_pkt_ratio", "service_bucket"
    ]

    train  = _combine(ctrl_train,  prot_train,  COLS)
    test   = _combine(ctrl_test,   prot_test,   COLS)
    cicids = _combine(ctrl_cicids, prot_cicids, COLS)

    # encode service_bucket AFTER combining
    # fixed categories ensures identical columns across all 3 datasets
    train  = encode_service_bucket(train)
    test   = encode_service_bucket(test)
    cicids = encode_service_bucket(cicids)

    _save(train, test, cicids,
          (KDD_TRAIN_EXP5, KDD_TEST_EXP5, CICIDS_EXP5),
          "Exp5 (+ service_bucket one-hot)")


# ════════════════════════════════════════
# RUN ALL AT ONCE
# ════════════════════════════════════════
def run_all_experiment_pipelines():
    print("[Pipeline] Building all experiment datasets...")
    run_exp1_pipeline()
    run_exp2_pipeline()
    run_exp3_pipeline()
    run_exp4_pipeline()
    run_exp5_pipeline()
    print("[Pipeline] All experiments done ✓")