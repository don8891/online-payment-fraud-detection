from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Preprocess IEEE-CIS numeric and categorical features.

    Numeric:
        Missing values -> median
        Scaling -> StandardScaler

    Categorical:
        Missing values -> most frequent value
        Encoding -> OneHotEncoder
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def build_model() -> RandomForestClassifier:
    """
    Build the baseline fraud detection model.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return model


def build_pipeline() -> Pipeline:
    """
    Complete machine learning pipeline:

    Raw IEEE-CIS features
            ↓
    Missing-value handling
            ↓
    Encoding + scaling
            ↓
    Random Forest
    """

    preprocessor = build_preprocessor()
    model = build_model()

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", model),
        ]
    )

    return pipeline