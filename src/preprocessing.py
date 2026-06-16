"""Preprocessing utilities for the diabetes classification project."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]
TARGET_COLUMN = "Outcome"

# Zero values are physically impossible for these columns in this dataset.
ZERO_AS_MISSING_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_data(data_path: Path | None = None) -> pd.DataFrame:
    root = get_project_root()
    path = data_path or root / "data" / "diabetes.csv"
    df = pd.read_csv(path, header=None, names=FEATURE_COLUMNS + [TARGET_COLUMN])
    return clean_data(df)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ZERO_AS_MISSING_COLUMNS:
        df[col] = df[col].replace(0, np.nan)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["GlucoseInsulinRatio"] = df["Glucose"] / (df["Insulin"] + 1)
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[20, 30, 40, 50, 60, 100],
        labels=["20-30", "30-40", "40-50", "50-60", "60+"],
    ).astype(str)
    df["BMICategory"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    ).astype(str)
    return df


def get_feature_columns(include_engineered: bool = True) -> list[str]:
    if include_engineered:
        return FEATURE_COLUMNS + ["GlucoseInsulinRatio", "AgeGroup", "BMICategory"]
    return FEATURE_COLUMNS.copy()


def build_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    numeric_features = [col for col in feature_columns if col not in {"AgeGroup", "BMICategory"}]
    categorical_features = [col for col in feature_columns if col in {"AgeGroup", "BMICategory"}]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    transformers = [("num", numeric_pipeline, numeric_features)]
    if categorical_features:
        from sklearn.preprocessing import OneHotEncoder

        categorical_pipeline.steps.append(
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers=transformers)
