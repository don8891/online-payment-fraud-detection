from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"

OUTPUT_FILE = PROCESSED_DIR / "ieee_fraud_dataset.csv"
TRAIN_FILE = PROCESSED_DIR / "ieee_train.csv"
TEST_FILE = PROCESSED_DIR / "ieee_test.csv"


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------
# MAIN DATA PREPARATION
# ---------------------------------------------------------

def prepare_ieee_data():

    print("Loading IEEE-CIS transaction data...")

    transactions = pd.read_csv(TRANSACTION_FILE)

    print(f"Transaction rows: {len(transactions):,}")
    print(f"Transaction columns: {len(transactions.columns):,}")

    print("\nLoading IEEE-CIS identity data...")

    identity = pd.read_csv(IDENTITY_FILE)

    print(f"Identity rows: {len(identity):,}")
    print(f"Identity columns: {len(identity.columns):,}")

    # -----------------------------------------------------
    # MERGE TRANSACTION + IDENTITY
    # -----------------------------------------------------

    print("\nMerging transaction and identity data...")

    df = transactions.merge(
        identity,
        on="TransactionID",
        how="left"
    )

    print(f"Merged rows: {len(df):,}")
    print(f"Merged columns: {len(df.columns):,}")

    # -----------------------------------------------------
    # BASIC CLEANING
    # -----------------------------------------------------

    print("\nCleaning data...")

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    print(f"Rows after cleaning: {len(df):,}")
    print(f"Columns after cleaning: {len(df.columns):,}")

    # -----------------------------------------------------
    # SAVE MERGED DATA
    # -----------------------------------------------------

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved merged dataset to:")
    print(OUTPUT_FILE)

    # -----------------------------------------------------
    # TRAIN / TEST SPLIT
    # -----------------------------------------------------

    print("\nCreating train/test split...")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["isFraud"]
    )

    train_df.to_csv(TRAIN_FILE, index=False)
    test_df.to_csv(TEST_FILE, index=False)

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")

    # -----------------------------------------------------
    # FRAUD DISTRIBUTION
    # -----------------------------------------------------

    fraud_rate = df["isFraud"].mean() * 100

    print(f"\nOverall fraud rate: {fraud_rate:.2f}%")

    print("\nFraud distribution:")
    print(df["isFraud"].value_counts())

    print("\nIEEE-CIS data preparation completed successfully.")


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------

if __name__ == "__main__":
    prepare_ieee_data()