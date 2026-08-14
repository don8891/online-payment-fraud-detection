from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .config import FEATURE_COLS, METRICS_DIR, MODELS_DIR
from .reason_codes import add_reason_codes
from .validation import DataValidationError, validate_scoring_dataframe


# IEEE-CIS model and threshold policy locations
MODEL_PATH = MODELS_DIR / "fraud_pipeline_ieee.joblib"
PRIMARY_THRESHOLD_PATH = METRICS_DIR / "threshold_policy.json"
FALLBACK_THRESHOLD_PATH = MODELS_DIR / "threshold_policy.json"


def load_model_and_threshold():
    """Load the trained IEEE-CIS fraud model and selected threshold policy."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"IEEE-CIS model not found at {MODEL_PATH}. "
            "Run `python -m src.train_model` first."
        )

    policy_path = PRIMARY_THRESHOLD_PATH
    if not policy_path.exists():
        policy_path = FALLBACK_THRESHOLD_PATH

    if not policy_path.exists():
        raise FileNotFoundError(
            f"IEEE-CIS threshold policy not found at {policy_path}. "
            "Run `python -m src.threshold_policy` first."
        )

    model = joblib.load(MODEL_PATH)

    with policy_path.open(encoding="utf-8") as f:
        policy = json.load(f)

    threshold_info = policy.get("recommended_threshold") or policy.get(
        "selected_threshold"
    )

    if isinstance(threshold_info, dict) and "threshold" in threshold_info:
        threshold = float(threshold_info["threshold"])
    elif isinstance(policy.get("threshold"), (int, float)):
        threshold = float(policy["threshold"])
    else:
        threshold = 0.60

    return model, threshold


def score_dataframe(
    df: pd.DataFrame,
    *,
    threshold: float | None = None,
    model=None,
) -> pd.DataFrame:
    """Score IEEE-CIS transactions and add fraud probability, decision, and reason codes."""

    if model is None or threshold is None:
        loaded_model, default_threshold = load_model_and_threshold()
        if model is None:
            model = loaded_model
        if threshold is None:
            threshold = default_threshold

    # Validate input dataset against required IEEE-CIS features
    validate_scoring_dataframe(df)

    df_scored = df.copy()

    # Extract required 18 IEEE-CIS feature columns
    features = df_scored[FEATURE_COLS]

    # Generate fraud probability
    probabilities = model.predict_proba(features)[:, 1]

    # Apply cost-sensitive threshold (default: 0.60)
    fraud_flags = (probabilities >= threshold).astype(int)

    df_scored["fraud_probability"] = probabilities
    df_scored["fraud_flag"] = fraud_flags

    # Human-readable decision
    df_scored["risk_level"] = df_scored["fraud_probability"].apply(
        lambda prob: "HIGH RISK" if prob >= threshold else "LOW RISK"
    )

    # Sort highest risk transactions to the top
    df_scored = df_scored.sort_values(
        "fraud_probability",
        ascending=False,
    ).reset_index(drop=True)

    # Add analyst-friendly reason codes
    df_scored = add_reason_codes(
        df_scored,
        threshold=threshold,
    )

    return df_scored


def score_file(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
) -> Path:
    """Score an IEEE-CIS transaction CSV and save the results."""

    model, threshold = load_model_and_threshold()
    input_csv = Path(input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    print(f"Loading transactions from: {input_csv}")
    df = pd.read_csv(input_csv)

    print(f"Transactions loaded: {len(df):,}")
    print(f"Decision threshold: {threshold:.2f}")

    df_scored = score_dataframe(
        df,
        threshold=threshold,
        model=model,
    )

    if output_csv is None:
        output_csv = input_csv.with_name(input_csv.stem + "_scored.csv")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df_scored.to_csv(
        output_csv,
        index=False,
    )

    flagged = int(df_scored["fraud_flag"].sum())

    print()
    print("=== SCORING SUMMARY ===")
    print(f"Rows scored:          {len(df_scored):,}")
    print(f"Fraud flagged:        {flagged:,}")
    print(
        f"Flagged rate:         "
        f"{flagged / len(df_scored):.2%}"
    )
    print(f"Output file:          {output_csv}")

    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score IEEE-CIS transactions for fraud risk."
    )

    parser.add_argument(
        "input_csv",
        type=str,
        help="Path to IEEE-CIS transaction CSV.",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output path for scored transactions.",
    )

    args = parser.parse_args()

    try:
        score_file(
            args.input_csv,
            args.output_csv,
        )
    except DataValidationError as exc:
        raise SystemExit(f"Input validation failed: {exc}") from exc


if __name__ == "__main__":
    main()