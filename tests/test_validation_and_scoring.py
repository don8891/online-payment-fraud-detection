from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.config import NUMERIC_FEATURES, TARGET_COL
from src.features import build_pipeline
from src.score_new_transactions import score_dataframe
from src.validation import (
    DataValidationError,
    validate_binary_target,
    validate_scoring_dataframe,
    validate_threshold,
    validate_training_dataframe,
)


class ValidationAndScoringTests(unittest.TestCase):
    def _demo_transactions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "TransactionID": range(1, 9),
                "TransactionDT": [86400, 86450, 86500, 86550, 86600, 86650, 86700, 86750],
                "TransactionAmt": [20.0, 450.0, 35.0, 800.0, 15.0, 900.0, 42.0, 1000.0],
                "card3": [150.0, 150.0, 185.0, 150.0, 150.0, 185.0, 150.0, 150.0],
                "card5": [226.0, 102.0, 137.0, 226.0, 102.0, 137.0, 226.0, 102.0],
                "addr1": [315.0, 476.0, 299.0, 315.0, 476.0, 299.0, 315.0, 476.0],
                "addr2": [87.0, 87.0, 87.0, 87.0, 87.0, 87.0, 87.0, 87.0],
                "C1": [1.0, 10.0, 2.0, 25.0, 1.0, 30.0, 3.0, 50.0],
                "C2": [1.0, 8.0, 2.0, 20.0, 1.0, 25.0, 2.0, 40.0],
                "C5": [0.0, 2.0, 0.0, 5.0, 0.0, 10.0, 1.0, 15.0],
                "C13": [1.0, 15.0, 3.0, 30.0, 1.0, 40.0, 2.0, 60.0],
                "D1": [14.0, 0.0, 10.0, 1.0, 30.0, 0.0, 5.0, 0.0],
                "D2": [14.0, 0.0, 10.0, 1.0, 30.0, 0.0, 5.0, 0.0],
                "ProductCD": ["W", "C", "W", "C", "W", "C", "W", "C"],
                "card4": ["visa", "mastercard", "visa", "mastercard", "visa", "mastercard", "visa", "mastercard"],
                "card6": ["debit", "credit", "debit", "credit", "debit", "credit", "debit", "credit"],
                "P_emaildomain": ["gmail.com", "anonymous.com", "yahoo.com", "anonymous.com", "gmail.com", "anonymous.com", "yahoo.com", "anonymous.com"],
                "R_emaildomain": ["gmail.com", "hotmail.com", "yahoo.com", "hotmail.com", "gmail.com", "hotmail.com", "yahoo.com", "hotmail.com"],
                "DeviceType": ["desktop", "mobile", "desktop", "mobile", "desktop", "mobile", "desktop", "mobile"],
                TARGET_COL: [0, 1, 0, 1, 0, 1, 0, 1],
            }
        )

    def test_training_dataframe_validation_accepts_valid_schema(self) -> None:
        df = self._demo_transactions()
        validate_training_dataframe(df)

    def test_scoring_dataframe_does_not_require_target(self) -> None:
        df = self._demo_transactions().drop(columns=[TARGET_COL])
        validate_scoring_dataframe(df)

    def test_missing_required_feature_raises_clear_error(self) -> None:
        df = self._demo_transactions().drop(columns=[NUMERIC_FEATURES[0]])
        with self.assertRaisesRegex(DataValidationError, "missing required columns"):
            validate_scoring_dataframe(df)

    def test_invalid_numeric_feature_raises(self) -> None:
        df = self._demo_transactions()
        df["TransactionAmt"] = df["TransactionAmt"].astype(object)
        df.loc[0, "TransactionAmt"] = "not-a-number"
        with self.assertRaisesRegex(DataValidationError, "invalid numeric"):
            validate_scoring_dataframe(df)

    def test_binary_target_validation_rejects_non_binary_labels(self) -> None:
        df = self._demo_transactions()
        df.loc[0, TARGET_COL] = 2
        with self.assertRaisesRegex(DataValidationError, "binary"):
            validate_binary_target(df)

    def test_threshold_validation_rejects_invalid_values(self) -> None:
        for value in [-0.1, 1.1]:
            with self.subTest(value=value), self.assertRaises(DataValidationError):
                validate_threshold(value)

    def test_score_dataframe_adds_probability_and_flag(self) -> None:
        df = self._demo_transactions()
        X = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL]

        model = build_pipeline()
        model.fit(X, y)

        scored = score_dataframe(df, threshold=0.5, model=model)

        self.assertIn("fraud_probability", scored.columns)
        self.assertIn("fraud_flag", scored.columns)
        self.assertIn("reason_codes", scored.columns)
        self.assertEqual(len(scored), len(df))
        self.assertTrue(scored["fraud_probability"].between(0, 1).all())
        self.assertTrue(set(scored["fraud_flag"].unique()).issubset({0, 1}))
        self.assertTrue(
            np.all(
                scored["fraud_probability"].to_numpy()[:-1]
                >= scored["fraud_probability"].to_numpy()[1:]
            )
        )


if __name__ == "__main__":
    unittest.main()

