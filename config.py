from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

#Data Paths

RAW_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
CLEANED_DIR = BASE_DIR / 'data' / 'cleaned'

KDD_RAW_TEST_PATH = RAW_DIR / 'KDDTest+.arff'
KDD_RAW_TRAIN_PATH = RAW_DIR / 'KDDTrain+.arff'

CICI_RAW_PATH = RAW_DIR / 'MachineLearningCSV' / 'MachineLearningCVE'
CICI_FRI_AFT_RAW_PATH = CICI_RAW_PATH / 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'


CICI_COMBINED_RAW_PATH = PROCESSED_DIR / 'cicids_combined.csv'

KDD_TRAIN_CONTROL = PROCESSED_DIR / 'kdd_train_control.csv'
KDD_TEST_CONTROL  = PROCESSED_DIR / 'kdd_test_control.csv'
CICIDS_CONTROL    = PROCESSED_DIR / 'cicids_control.csv'

PROTOCOL_AWARE_KDD_TRAIN = PROCESSED_DIR / 'kdd_train_protocol_aware.csv'
PROTOCOL_AWARE_KDD_TEST  = PROCESSED_DIR / 'kdd_test_protocol_aware.csv'
PROTOCOL_AWARE_CICIDS    = PROCESSED_DIR / 'cicids_protocol_aware.csv'

# Experiment CSVs
KDD_TRAIN_EXP1  = PROCESSED_DIR / 'kdd_train_exp1.csv'
KDD_TEST_EXP1   = PROCESSED_DIR / 'kdd_test_exp1.csv'
CICIDS_EXP1     = PROCESSED_DIR / 'cicids_exp1.csv'

KDD_TRAIN_EXP2  = PROCESSED_DIR / 'kdd_train_exp2.csv'
KDD_TEST_EXP2   = PROCESSED_DIR / 'kdd_test_exp2.csv'
CICIDS_EXP2     = PROCESSED_DIR / 'cicids_exp2.csv'

KDD_TRAIN_EXP3  = PROCESSED_DIR / 'kdd_train_exp3.csv'
KDD_TEST_EXP3   = PROCESSED_DIR / 'kdd_test_exp3.csv'
CICIDS_EXP3     = PROCESSED_DIR / 'cicids_exp3.csv'

KDD_TRAIN_EXP4  = PROCESSED_DIR / 'kdd_train_exp4.csv'
KDD_TEST_EXP4   = PROCESSED_DIR / 'kdd_test_exp4.csv'
CICIDS_EXP4     = PROCESSED_DIR / 'cicids_exp4.csv'

KDD_TRAIN_EXP5  = PROCESSED_DIR / 'kdd_train_exp5.csv'
KDD_TEST_EXP5   = PROCESSED_DIR / 'kdd_test_exp5.csv'
CICIDS_EXP5     = PROCESSED_DIR / 'cicids_exp5.csv'

# Label paths (needed to attach labels to experiment CSVs)
KDD_TRAIN_LABELS = PROCESSED_DIR / 'kdd_train_labels.csv'
KDD_TEST_LABELS  = PROCESSED_DIR / 'kdd_test_labels.csv'
CICIDS_LABELS    = PROCESSED_DIR / 'cicids_labels.csv'