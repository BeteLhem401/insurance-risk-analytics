
"""
modeling.py

Reusable modeling utilities for insurance risk analytics.
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None


def engineer_features(df):
    df = df.copy()

    current_year = 2026

    if "RegistrationYear" in df.columns:
        df["VehicleAge"] = current_year - df["RegistrationYear"]

    df["HasClaim"] = np.where(df["TotalClaims"] > 0, 1, 0)

    df["LossRatio"] = (
        df["TotalClaims"] / df["TotalPremium"].replace(0, np.nan)
    )

    df["Margin"] = df["TotalPremium"] - df["TotalClaims"]

    return df


def prepare_regression_data(df, target="TotalClaims"):
    df = engineer_features(df)

    df = df[df["TotalClaims"] > 0]

    drop_cols = [
        target,
        "PolicyID",
        "UnderwrittenCoverID"
    ]

    features = [c for c in df.columns if c not in drop_cols]

    X = df[features]
    y = df[target]

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numerical_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_cols,
            ),
        ]
    )

    return X, y, preprocessor


def train_models(X_train, y_train, preprocessor):

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    if XGBRegressor:
        models["XGBoost"] = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )

    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        trained_models[name] = pipeline

    return trained_models


def evaluate_regression_models(models, X_test, y_test):

    results = []

    for name, model in models.items():

        preds = model.predict(X_test)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        r2 = r2_score(y_test, preds)

        results.append({
            "Model": name,
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        })

    return pd.DataFrame(results)


def train_claim_classifier(df):

    df = engineer_features(df)

    target = "HasClaim"

    drop_cols = [
        target,
        "PolicyID",
        "UnderwrittenCoverID"
    ]

    features = [c for c in df.columns if c not in drop_cols]

    X = df[features]
    y = df[target]

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                numerical_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_cols,
            ),
        ]
    )

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    metrics = {
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "F1": round(f1_score(y_test, preds), 4),
    }

    return clf, metrics


def calculate_risk_based_premium(
    claim_probability,
    predicted_severity,
    expense_loading=500,
    profit_margin=0.1
):

    premium = (
        (claim_probability * predicted_severity)
        + expense_loading
    )

    premium = premium * (1 + profit_margin)

    return premium
