from __future__ import annotations

import json

import joblib
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)

from .config import (
    METRICS_DIR,
    MODELS_DIR,
    TARGET_COL,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    FEATURE_COLS,
)

from .features import build_pipeline


def load_processed_data():
    """
    Load the prepared IEEE-CIS train/test datasets.
    """

    if not TRAIN_DATA_PATH.exists() or not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "IEEE-CIS processed train/test files were not found.\n"
            "Run:\n"
            "python -m src.prepare_ieee_data"
        )

    print("Loading IEEE-CIS training data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)

    print("Loading IEEE-CIS testing data...")
    test_df = pd.read_csv(TEST_DATA_PATH)

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows:  {len(test_df):,}")

    return train_df, test_df


def train_and_evaluate() -> dict:
    """
    Train the baseline IEEE-CIS fraud detection model
    and evaluate its performance.
    """

    train_df, test_df = load_processed_data()

    # -----------------------------------------------------
    # SELECT FEATURES
    # -----------------------------------------------------

    print("\nSelecting IEEE-CIS features...")

    missing_train = [
        col for col in FEATURE_COLS
        if col not in train_df.columns
    ]

    missing_test = [
        col for col in FEATURE_COLS
        if col not in test_df.columns
    ]

    if missing_train or missing_test:
        raise ValueError(
            f"Missing features.\n"
            f"Train: {missing_train}\n"
            f"Test: {missing_test}"
        )

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    print(f"Number of features: {len(FEATURE_COLS)}")

    print(f"Fraud rate in training data: {y_train.mean():.4%}")
    print(f"Fraud rate in testing data:  {y_test.mean():.4%}")

    # -----------------------------------------------------
    # BUILD MODEL
    # -----------------------------------------------------

    print("\nBuilding Random Forest pipeline...")

    pipeline = build_pipeline()

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    print("Training model...")

    pipeline.fit(X_train, y_train)

    print("Training completed.")

    # -----------------------------------------------------
    # PREDICTIONS
    # -----------------------------------------------------

    print("\nGenerating fraud probabilities...")

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Default threshold for baseline
    y_pred_default = (y_proba >= 0.5).astype(int)

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    roc_auc = roc_auc_score(
        y_test,
        y_proba,
    )

    average_precision = average_precision_score(
        y_test,
        y_proba,
    )

    brier_score = brier_score_loss(
        y_test,
        y_proba,
    )

    cls_report = classification_report(
        y_test,
        y_pred_default,
        output_dict=True,
        digits=3,
    )

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    metrics = {
        "dataset": "IEEE-CIS Fraud Detection Dataset",
        "model": "RandomForestClassifier",
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "brier_score": float(brier_score),
        "classification_report_default_threshold": cls_report,
        "n_train_samples": int(len(y_train)),
        "n_test_samples": int(len(y_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "features": FEATURE_COLS,
    }

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    model_path = MODELS_DIR / "fraud_pipeline_ieee.joblib"

    joblib.dump(
        pipeline,
        model_path,
    )

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    metrics_path = METRICS_DIR / "ieee_metrics.json"

    with metrics_path.open("w") as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # DISPLAY RESULTS
    # -----------------------------------------------------

    print("\n========================================")
    print("IEEE-CIS MODEL RESULTS")
    print("========================================")

    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Average Precision: {average_precision:.4f}")
    print(f"Brier Score:       {brier_score:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred_default,
            digits=3,
        )
    )

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")

    return metrics


def main() -> None:

    metrics = train_and_evaluate()

    print("\n=== Training completed successfully ===")

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()  