from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    FEATURE_COLS,
    METRICS_DIR,
    MODELS_DIR,
    TARGET_COL,
    TEST_DATA_PATH,
    THRESHOLD_GRID,
)


# =========================================================
# LOAD MODEL AND TEST DATA
# =========================================================

def load_model_and_test_data():
    """
    Load the trained IEEE-CIS fraud model and test dataset.
    """

    model_path = MODELS_DIR / "fraud_pipeline_ieee.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Run: python -m src.train_model"
        )

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found: {TEST_DATA_PATH}\n"
            "Run: python -m src.prepare_ieee_data"
        )

    print("Loading trained IEEE-CIS model...")
    model = joblib.load(model_path)

    print("Loading IEEE-CIS test data...")
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    return model, X_test, y_test


# =========================================================
# THRESHOLD SEARCH
# =========================================================

def search_thresholds(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> list[dict]:
    """
    Test multiple probability thresholds and calculate
    fraud detection performance and business cost.
    """

    print("\nGenerating fraud probabilities...")

    probabilities = model.predict_proba(X_test)[:, 1]

    results = []

    print("\nSearching decision thresholds...")

    for threshold in THRESHOLD_GRID:

        # Convert probability into fraud/not-fraud decision
        y_pred = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0,
        )

        # False Positive Rate
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # Percentage of transactions sent for analyst review
        flagged_rate = (tp + fp) / len(y_test)

        # Business cost
        #
        # False Positive:
        # Legitimate transaction incorrectly flagged.
        #
        # False Negative:
        # Fraudulent transaction incorrectly classified
        # as legitimate.

        cost = (
            fp * COST_FALSE_POSITIVE
            + fn * COST_FALSE_NEGATIVE
        )

        # Normalize cost so it is easier to compare
        normalized_cost = cost / len(y_test)

        results.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "false_positive_rate": float(false_positive_rate),
                "specificity": float(specificity),
                "flagged_rate": float(flagged_rate),
                "cost": float(cost),
                "normalized_cost": float(normalized_cost),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn),
            }
        )

    return results


# =========================================================
# FIND BEST THRESHOLD
# =========================================================

def find_best_threshold(
    threshold_results: list[dict],
) -> dict:
    """
    Select the threshold with the lowest business cost.

    If two thresholds have the same cost, prefer the one
    with higher recall.
    """

    best = min(
        threshold_results,
        key=lambda row: (
            row["cost"],
            -row["recall"],
        ),
    )

    return best


def build_threshold_policy_candidates(
    threshold_results: list[dict],
) -> list[dict]:
    """Build key threshold policy candidate options."""

    if not threshold_results:
        return []

    cost_opt = min(
        threshold_results,
        key=lambda r: (r["cost"], -r["recall"]),
    )

    bal_f1 = max(
        threshold_results,
        key=lambda r: (r["f1"], r["recall"]),
    )

    hi_rec = max(
        threshold_results,
        key=lambda r: (r["recall"], -r["cost"]),
    )

    hi_prec = max(
        threshold_results,
        key=lambda r: (r["precision"], r["recall"]),
    )

    rev_cap = min(
        threshold_results,
        key=lambda r: (abs(r["flagged_rate"] - 0.10), r["cost"]),
    )

    return [
        {
            **cost_opt,
            "policy": "cost_optimized",
            "rationale": "Minimizes total business cost under given cost weights",
        },
        {
            **bal_f1,
            "policy": "balanced_f1",
            "rationale": "Maximizes harmonic mean of precision and recall",
        },
        {
            **hi_rec,
            "policy": "high_recall",
            "rationale": "Prioritizes capturing maximum fraud cases",
        },
        {
            **hi_prec,
            "policy": "high_precision",
            "rationale": "Minimizes false positive review volume",
        },
        {
            **rev_cap,
            "policy": "review_capacity",
            "rationale": "Target review capacity around 10% volume",
        },
    ]


def build_threshold_policy_summary(
    threshold_results: list[dict],
    best_threshold: dict,
    cost_false_positive: float = COST_FALSE_POSITIVE,
    cost_false_negative: float = COST_FALSE_NEGATIVE,
) -> dict:
    """Build full summary of threshold policy and candidates."""

    candidates = build_threshold_policy_candidates(threshold_results)

    return {
        "purpose": (
            "Cost-sensitive decision threshold for "
            "IEEE-CIS fraud-risk analyst review."
        ),
        "recommended_policy": "cost_optimized",
        "recommended_threshold": best_threshold,
        "cost_assumptions": {
            "false_positive_cost": float(cost_false_positive),
            "false_negative_cost": float(cost_false_negative),
        },
        "policy_candidates": candidates,
        "threshold_results": threshold_results,
        "interpretation": (
            "A transaction is flagged for analyst review when "
            "its predicted fraud probability is greater than "
            "or equal to the selected threshold."
        ),
    }


def save_threshold_policy_artifacts(
    threshold_results: list[dict],
    best_threshold: dict,
    metrics_dir: Path | None = None,
    cost_false_positive: float = COST_FALSE_POSITIVE,
    cost_false_negative: float = COST_FALSE_NEGATIVE,
) -> dict[str, Path]:
    """Save policy JSON, CSV, and Markdown documentation."""

    if metrics_dir is None:
        metrics_dir = METRICS_DIR

    metrics_dir = Path(metrics_dir)
    metrics_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = build_threshold_policy_summary(
        threshold_results,
        best_threshold,
        cost_false_positive=cost_false_positive,
        cost_false_negative=cost_false_negative,
    )

    json_path = metrics_dir / "threshold_policy.json"
    with json_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )

    csv_path = metrics_dir / "threshold_results.csv"
    if threshold_results:
        fieldnames = list(threshold_results[0].keys())
        with csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            writer.writerows(threshold_results)

    md_path = metrics_dir / "threshold_policy.md"
    lines = [
        "# IEEE-CIS Fraud Threshold Policy",
        "",
        "## Recommended threshold",
        "",
        f"**Threshold:** `{best_threshold['threshold']:.2f}`",
        "",
        "## Policy candidates",
        "",
    ]

    for candidate in summary["policy_candidates"]:
        lines.append(
            f"- **{candidate['policy']}**: threshold `{candidate['threshold']:.2f}` ({candidate['rationale']})"
        )

    lines.extend(
        [
            "",
            "## Cost assumptions",
            "",
            f"- False positive cost: `{cost_false_positive}`",
            f"- False negative cost: `{cost_false_negative}`",
            "",
        ]
    )

    md_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": md_path,
    }


def save_policy(
    threshold_results: list[dict],
    best_threshold: dict,
) -> None:
    """Save threshold search results and the selected policy."""

    artifacts = save_threshold_policy_artifacts(
        threshold_results,
        best_threshold,
    )

    print("\nPolicy artifacts saved:")
    print(f"JSON:     {artifacts['json']}")
    print(f"CSV:      {artifacts['csv']}")
    print(f"Markdown: {artifacts['markdown']}")



# =========================================================
# DISPLAY RESULTS
# =========================================================

def display_results(
    threshold_results: list[dict],
    best_threshold: dict,
) -> None:

    print("\n")
    print("=" * 60)
    print("IEEE-CIS COST-SENSITIVE THRESHOLD ANALYSIS")
    print("=" * 60)

    print(
        f"\nRecommended threshold: "
        f"{best_threshold['threshold']:.2f}"
    )

    print("\nPerformance at recommended threshold:")

    print(
        f"Precision:           "
        f"{best_threshold['precision']:.4f}"
    )

    print(
        f"Recall:              "
        f"{best_threshold['recall']:.4f}"
    )

    print(
        f"F1 Score:            "
        f"{best_threshold['f1']:.4f}"
    )

    print(
        f"False Positive Rate: "
        f"{best_threshold['false_positive_rate']:.4f}"
    )

    print(
        f"Flagged for Review:  "
        f"{best_threshold['flagged_rate']:.4f}"
    )

    print(
        f"True Positives:      "
        f"{best_threshold['tp']:,}"
    )

    print(
        f"False Positives:     "
        f"{best_threshold['fp']:,}"
    )

    print(
        f"True Negatives:      "
        f"{best_threshold['tn']:,}"
    )

    print(
        f"False Negatives:     "
        f"{best_threshold['fn']:,}"
    )

    print(
        f"\nBusiness Cost:       "
        f"{best_threshold['cost']:,.0f}"
    )

    print(
        f"Normalized Cost:     "
        f"{best_threshold['normalized_cost']:.6f}"
    )

    # -----------------------------------------------------
    # Show threshold comparison
    # -----------------------------------------------------

    print("\nThreshold comparison:")
    print("-" * 60)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Cost':<12}"
    )

    print("-" * 60)

    for row in threshold_results:

        print(
            f"{row['threshold']:<12.2f}"
            f"{row['precision']:<12.3f}"
            f"{row['recall']:<12.3f}"
            f"{row['f1']:<12.3f}"
            f"{row['cost']:<12.0f}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    print("=" * 60)
    print("IEEE-CIS FRAUD THRESHOLD OPTIMIZATION")
    print("=" * 60)

    # Load trained model and test data
    model, X_test, y_test = load_model_and_test_data()

    # Search thresholds
    threshold_results = search_thresholds(
        model,
        X_test,
        y_test,
    )

    # Find cost-optimal threshold
    best_threshold = find_best_threshold(
        threshold_results
    )

    # Display results
    display_results(
        threshold_results,
        best_threshold,
    )

    # Save artifacts
    save_policy(
        threshold_results,
        best_threshold,
    )

    print("\nThreshold optimization completed successfully.")


if __name__ == "__main__":
    main()