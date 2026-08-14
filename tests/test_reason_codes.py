from __future__ import annotations

import unittest

import pandas as pd

from src.reason_codes import (
    add_reason_codes,
    high_amount_cutoff,
    humanize_feature_name,
    reason_codes_for_row,
    shap_reason_codes,
    split_reason_codes,
)


class ReasonCodeTests(unittest.TestCase):
    def _demo_scored(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "TransactionDT": [86400, 86450, 86500],
                "TransactionAmt": [25.0, 900.0, 80.0],
                "card3": [150.0, 185.0, 150.0],
                "card5": [226.0, 137.0, 102.0],
                "addr1": [315.0, 476.0, 299.0],
                "addr2": [87.0, 87.0, 87.0],
                "C1": [1.0, 25.0, 2.0],
                "C2": [1.0, 20.0, 2.0],
                "C5": [0.0, 5.0, 0.0],
                "C13": [1.0, 30.0, 3.0],
                "D1": [14.0, 0.0, 10.0],
                "D2": [14.0, 0.0, 10.0],
                "ProductCD": ["W", "C", "W"],
                "card4": ["visa", "mastercard", "visa"],
                "card6": ["debit", "credit", "debit"],
                "P_emaildomain": ["gmail.com", "anonymous.com", "yahoo.com"],
                "R_emaildomain": ["gmail.com", "hotmail.com", "yahoo.com"],
                "DeviceType": ["desktop", "mobile", "desktop"],
                "fraud_probability": [0.05, 0.97, 0.55],
                "fraud_flag": [0, 1, 0],
            }
        )

    def test_high_amount_cutoff_uses_batch_quantile(self) -> None:
        df = self._demo_scored()
        cutoff = high_amount_cutoff(df, quantile=0.50)
        self.assertEqual(cutoff, 80.0)

    def test_reason_codes_for_high_risk_row_are_analyst_friendly(self) -> None:
        row = self._demo_scored().iloc[1]
        reasons = reason_codes_for_row(
            row,
            threshold=0.60,
            amount_cutoff=800.0,
            max_reasons=10,
        )

        joined = " | ".join(reasons).lower()
        self.assertIn("very high model fraud probability", joined)
        self.assertIn("transaction amount is unusually high", joined)
        self.assertIn("high card transaction activity", joined)

    def test_add_reason_codes_adds_string_column(self) -> None:
        result = add_reason_codes(self._demo_scored(), threshold=0.60)

        self.assertIn("reason_codes", result.columns)
        self.assertEqual(len(result), 3)
        self.assertTrue(result["reason_codes"].str.len().gt(0).all())

    def test_humanize_feature_name_handles_transformed_names(self) -> None:
        self.assertEqual(
            humanize_feature_name("numeric__TransactionAmt"),
            "transaction amount",
        )
        self.assertEqual(
            humanize_feature_name("categorical__ProductCD_C"),
            "product category = C",
        )

    def test_shap_reason_codes_use_direction(self) -> None:
        reasons = shap_reason_codes(
            [0.8, -0.4, 0.1],
            [
                "numeric__TransactionAmt",
                "categorical__ProductCD_C",
                "numeric__C13",
            ],
            max_reasons=2,
        )

        self.assertEqual(len(reasons), 2)
        self.assertIn("transaction amount increased fraud risk", reasons[0])
        self.assertIn("product category = C reduced fraud risk", reasons[1])

    def test_split_reason_codes_handles_empty_values(self) -> None:
        self.assertEqual(split_reason_codes(None), [])
        self.assertEqual(
            split_reason_codes("High risk; Large amount"), ["High risk", "Large amount"]
        )


if __name__ == "__main__":
    unittest.main()

