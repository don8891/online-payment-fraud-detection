from pathlib import Path


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# IEEE-CIS raw datasets
TRANSACTION_DATA_PATH = RAW_DATA_DIR / "train_transaction.csv"
IDENTITY_DATA_PATH = RAW_DATA_DIR / "train_identity.csv"

# Processed IEEE-CIS datasets
IEEE_DATA_PATH = PROCESSED_DATA_DIR / "ieee_fraud_dataset.csv"
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "ieee_train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "ieee_test.csv"

MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIGURES_DIR = REPORTS_DIR / "figures"


# Create directories if they don't exist

for directory in [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    METRICS_DIR,
    FIGURES_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# =========================================================
# TARGET
# =========================================================

TARGET_COL = "isFraud"


# =========================================================
# IDENTIFIERS
# =========================================================

# These identify a transaction but should NOT be used
# as predictive features.

ID_COLS = [
    "TransactionID",
]


# =========================================================
# IEEE-CIS CATEGORICAL FEATURES
# =========================================================

CATEGORICAL_FEATURES = [
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
]


# =========================================================
# IEEE-CIS NUMERIC FEATURES
# =========================================================

NUMERIC_FEATURES = [
    "TransactionDT",
    "TransactionAmt",
    "card3",
    "card5",
    "addr1",
    "addr2",
    "C1",
    "C2",
    "C5",
    "C13",
    "D1",
    "D2",
]


# =========================================================
# ALL FEATURES
# =========================================================

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# =========================================================
# TRAIN / TEST CONFIGURATION
# =========================================================

TEST_SIZE = 0.20

RANDOM_STATE = 42


# =========================================================
# THRESHOLD SEARCH
# =========================================================

THRESHOLD_GRID = [
    i / 100 for i in range(5, 100, 5)
]


# =========================================================
# COST-SENSITIVE FRAUD DETECTION
# =========================================================

# Missing a fraudulent transaction is considered
# more expensive than incorrectly flagging a legitimate one.

COST_FALSE_NEGATIVE = 10.0

COST_FALSE_POSITIVE = 1.0