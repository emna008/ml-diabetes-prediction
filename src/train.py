"""Model training and evaluation utilities."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocessing import (
    TARGET_COLUMN,
    add_engineered_features,
    build_preprocessor,
    get_feature_columns,
    load_data,
)


def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


def evaluate_model(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def train_and_compare(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, Pipeline, dict]:
    df = add_engineered_features(load_data())
    feature_columns = get_feature_columns(include_engineered=True)
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(feature_columns)
    results = []
    best_pipeline = None
    best_f1 = -1.0
    fitted_models = {}

    for name, model in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate_model(y_test, y_pred, y_proba)
        cv_scores = cross_val_score(
            pipeline, X, y, cv=5, scoring="f1", n_jobs=-1
        )
        metrics["cv_f1_mean"] = cv_scores.mean()
        metrics["cv_f1_std"] = cv_scores.std()
        metrics["model"] = name
        results.append(metrics)
        fitted_models[name] = pipeline

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_pipeline = pipeline

    results_df = pd.DataFrame(results).set_index("model")
    return results_df, best_pipeline, {
        "X_test": X_test,
        "y_test": y_test,
        "fitted_models": fitted_models,
    }


def tune_random_forest(X_train, y_train, preprocessor) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(random_state=42),
            ),
        ]
    )
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
    }
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def save_best_model(model: Pipeline, path: Path | None = None) -> Path:
    root = Path(__file__).resolve().parent.parent
    output = path or root / "models" / "best_model.joblib"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    return output


if __name__ == "__main__":
    results_df, best_model, _ = train_and_compare()
    print(results_df.round(4))
    model_path = save_best_model(best_model)
    print(f"Best model saved to {model_path}")
