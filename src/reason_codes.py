from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


# Features used by the IEEE-CIS Random Forest model.
IEEE_FEATURES = [
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
    "ProductCD",
    "card4",
    "card6",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
]


FRIENDLY_FEATURE_NAMES = {
    "TransactionDT": "transaction time",
    "TransactionAmt": "transaction amount",
    "card3": "card type indicator",
    "card5": "card issuer indicator",
    "addr1": "billing address",
    "addr2": "billing region",
    "C1": "card transaction count",
    "C2": "card activity count",
    "C5": "transaction activity count",
    "C13": "transaction frequency indicator",
    "D1": "days since previous transaction",
    "D2": "days since previous transaction activity",
    "ProductCD": "product category",
    "card4": "card network",
    "card6": "card type",
    "P_emaildomain": "purchaser email domain",
    "R_emaildomain": "recipient email domain",
    "DeviceType": "device type",
}


def _safe_float(
    value,
    default: float | None = None,
) -> float | None:
    """Safely convert a value to float."""

    try:
        if value is None or pd.isna(value):
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def _safe_str(value) -> str:
    """Safely convert a value to string."""

    if value is None or pd.isna(value):
        return "unknown"

    return str(value)


def high_amount_cutoff(
    df: pd.DataFrame,
    quantile: float = 0.95,
) -> float | None:
    """Calculate a high transaction amount cutoff for the current batch."""

    if "TransactionAmt" not in df.columns or len(df) == 0:
        return None

    values = pd.to_numeric(
        df["TransactionAmt"],
        errors="coerce",
    ).dropna()

    if len(values) == 0:
        return None

    return float(values.quantile(quantile))


def reason_codes_for_row(
    row: pd.Series | dict,
    *,
    threshold: float = 0.60,
    amount_cutoff: float | None = None,
    max_reasons: int = 5,
) -> list[str]:
    """
    Generate deterministic analyst-friendly reason codes
    for an IEEE-CIS transaction.

    These are heuristic risk indicators, not causal explanations.
    SHAP will later provide model-based explanations.
    """

    reasons: list[str] = []

    # ---------------------------------------------------------
    # Model risk
    # ---------------------------------------------------------

    fraud_probability = _safe_float(
        row.get("fraud_probability")
    )

    if fraud_probability is not None:

        if fraud_probability >= 0.85:
            reasons.append(
                "Very high model fraud probability"
            )

        elif fraud_probability >= threshold:
            reasons.append(
                "Model probability is above the review threshold"
            )

    # ---------------------------------------------------------
    # Transaction amount
    # ---------------------------------------------------------

    amount = _safe_float(
        row.get("TransactionAmt")
    )

    if (
        amount is not None
        and amount_cutoff is not None
        and amount >= amount_cutoff
    ):
        reasons.append(
            "Transaction amount is unusually high"
        )

    # ---------------------------------------------------------
    # Product category
    # ---------------------------------------------------------

    product = _safe_str(
        row.get("ProductCD")
    ).upper()

    if product != "UNKNOWN":
        reasons.append(
            f"Product category is {product}"
        )

    # ---------------------------------------------------------
    # Card information
    # ---------------------------------------------------------

    card_network = _safe_str(
        row.get("card4")
    ).lower()

    if card_network != "unknown":
        reasons.append(
            f"Card network is {card_network}"
        )

    card_type = _safe_str(
        row.get("card6")
    ).lower()

    if card_type != "unknown":
        reasons.append(
            f"Card type is {card_type}"
        )

    # ---------------------------------------------------------
    # Email information
    # ---------------------------------------------------------

    purchaser_email = _safe_str(
        row.get("P_emaildomain")
    ).lower()

    recipient_email = _safe_str(
        row.get("R_emaildomain")
    ).lower()

    if purchaser_email != "unknown":
        reasons.append(
            f"Purchaser email domain is {purchaser_email}"
        )

    if (
        recipient_email != "unknown"
        and recipient_email != purchaser_email
    ):
        reasons.append(
            "Purchaser and recipient email domains differ"
        )

    # ---------------------------------------------------------
    # Device information
    # ---------------------------------------------------------

    device_type = _safe_str(
        row.get("DeviceType")
    ).lower()

    if device_type != "unknown":
        reasons.append(
            f"Transaction originated from {device_type} device"
        )

    # ---------------------------------------------------------
    # Address information
    # ---------------------------------------------------------

    addr1 = _safe_str(
        row.get("addr1")
    )

    addr2 = _safe_str(
        row.get("addr2")
    )

    if addr1 != "unknown" and addr2 != "unknown":
        reasons.append(
            "Billing address information is available"
        )

    # ---------------------------------------------------------
    # Activity indicators
    # ---------------------------------------------------------

    c1 = _safe_float(row.get("C1"))
    c13 = _safe_float(row.get("C13"))

    if c1 is not None and c1 > 20:
        reasons.append(
            "High card transaction activity"
        )

    if c13 is not None and c13 > 20:
        reasons.append(
            "High transaction frequency indicator"
        )

    # ---------------------------------------------------------
    # Temporal indicators
    # ---------------------------------------------------------

    d1 = _safe_float(row.get("D1"))
    d2 = _safe_float(row.get("D2"))

    if d1 is not None and d1 <= 1:
        reasons.append(
            "Transaction is close to a previous transaction in time"
        )

    if d2 is not None and d2 <= 1:
        reasons.append(
            "Recent transaction activity detected"
        )

    # ---------------------------------------------------------
    # Fallback
    # ---------------------------------------------------------

    if not reasons:
        reasons.append(
            "No strong rule-based risk indicators identified"
        )

    return reasons[:max_reasons]


def add_reason_codes(
    df_scored: pd.DataFrame,
    *,
    threshold: float = 0.60,
    max_reasons: int = 5,
) -> pd.DataFrame:
    """Add reason codes to scored IEEE-CIS transactions."""

    df = df_scored.copy()

    cutoff = high_amount_cutoff(df)

    df["reason_codes"] = [
        "; ".join(
            reason_codes_for_row(
                record,
                threshold=threshold,
                amount_cutoff=cutoff,
                max_reasons=max_reasons,
            )
        )
        for record in df.to_dict("records")
    ]

    return df


def humanize_feature_name(
    feature_name: str,
) -> str:
    """
    Convert IEEE-CIS model feature names into
    analyst-friendly names.
    """

    raw = str(feature_name)

    # Remove sklearn transformer prefixes.
    for prefix in (
        "numeric__",
        "categorical__",
        "num__",
        "cat__",
    ):
        if raw.startswith(prefix):
            raw = raw[len(prefix):]
            break

    # Handle one-hot encoded categorical features.
    for base in (
        "ProductCD",
        "card4",
        "card6",
        "P_emaildomain",
        "R_emaildomain",
        "DeviceType",
    ):
        if raw.startswith(base + "_"):

            value = raw[
                len(base) + 1:
            ].replace("_", " ")

            friendly_base = FRIENDLY_FEATURE_NAMES.get(
                base,
                base.replace("_", " "),
            )

            return (
                f"{friendly_base} = {value}"
            )

    return FRIENDLY_FEATURE_NAMES.get(
        raw,
        raw.replace("_", " "),
    )


def positive_class_shap_values(
    shap_values: object,
) -> np.ndarray:
    """
    Normalize SHAP output across SHAP versions.

    Supports:
    - legacy list output
    - modern 3-D output
    - already 2-D output
    """

    if isinstance(shap_values, list):

        return np.asarray(
            shap_values[1]
        )

    arr = np.asarray(
        shap_values
    )

    if arr.ndim == 3:

        return arr[..., 1]

    return arr


def shap_reason_codes(
    shap_values: Iterable[float],
    feature_names: Iterable[str],
    *,
    max_reasons: int = 5,
) -> list[str]:
    """
    Convert SHAP feature contributions into
    analyst-friendly explanations.
    """

    values = np.asarray(
        list(shap_values),
        dtype=float,
    ).reshape(-1)

    names = np.asarray(
        list(feature_names),
        dtype=object,
    ).reshape(-1)

    n = min(
        len(values),
        len(names),
    )

    if n == 0:
        return [
            "No SHAP reason codes available"
        ]

    values = values[:n]
    names = names[:n]

    # Sort by absolute SHAP contribution.
    order = np.argsort(
        np.abs(values)
    )[::-1]

    reasons: list[str] = []

    for idx in order:

        value = float(
            values[idx]
        )

        if np.isclose(
            value,
            0.0,
        ):
            continue

        feature = humanize_feature_name(
            str(names[idx])
        )

        if value > 0:
            reasons.append(
                f"{feature} increased fraud risk"
            )
        else:
            reasons.append(
                f"{feature} reduced fraud risk"
            )

        if len(reasons) >= max_reasons:
            break

    if not reasons:
        return [
            "No strong SHAP drivers identified"
        ]

    return reasons


def split_reason_codes(
    value: str | float | None,
) -> list[str]:
    """Convert saved reason-code text into a list."""

    if value is None or pd.isna(value):
        return []

    return [
        item.strip()
        for item in str(value).split(";")
        if item.strip()
    ]