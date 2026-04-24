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