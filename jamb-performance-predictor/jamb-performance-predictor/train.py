"""Train the pass/fail model used by the web app.

Pass = JAMB score >= 200. Follows the thesis pipeline (Chapter 3 and Appendix):
impute -> one-hot encode -> standardise -> 80/20 split -> Logistic Regression.

Run:  python train.py
Writes: logreg_model.pkl, scaler.pkl, feature_columns.pkl, model_meta.json
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from project_info import PASS_MARK

DATA_PATH = Path("dataset/jamb_exam_results.csv")
TARGET = "JAMB_Score"
ID_COLUMN = "Student_ID"
ARTIFACTS = ("logreg_model.pkl", "scaler.pkl", "feature_columns.pkl", "model_meta.json")


def load_dataset(path=DATA_PATH) -> pd.DataFrame:
    """Read the Kaggle CSV and normalise the target column name."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{path}'. Download jamb_exam_results.csv from Kaggle "
            "(see dataset/README.md) and place it there."
        )
    df = pd.read_csv(path)
    renamed = {c: TARGET for c in df.columns if c.lower() == TARGET.lower()}
    df = df.rename(columns=renamed)
    if TARGET not in df.columns:
        raise ValueError(f"Column '{TARGET}' not found. Columns present: {list(df.columns)}")
    return df


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Mean for numeric columns, mode for categorical ones (thesis section 3.2.2)."""
    df = df.copy()
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df


def train(data_path=DATA_PATH, out_dir=".") -> dict:
    out_dir = Path(out_dir)
    df = impute(load_dataset(data_path))

    y = (df[TARGET] >= PASS_MARK).astype(int)
    X = df.drop(columns=[c for c in (TARGET, ID_COLUMN) if c in df.columns])

    num_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_features = [c for c in X.columns if c not in num_features]

    X_encoded = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # Fit the scaler on the training split only, so the test set stays unseen.
    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.transform(X_train), y_train)

    pred = model.predict(scaler.transform(X_test))
    proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else None,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
    }

    meta = {
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "pass_mark": PASS_MARK,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "pass_rate": float(y.mean()),
        "metrics": metrics,
        "categorical": {
            c: {
                "options": sorted(X[c].astype(str).unique().tolist()),
                "mode": str(X[c].mode().iloc[0]),
            }
            for c in cat_features
        },
        "numeric": {
            c: {
                "min": float(X[c].min()),
                "max": float(X[c].max()),
                "median": float(X[c].median()),
                "is_int": bool(pd.api.types.is_integer_dtype(df[c])),
            }
            for c in num_features
        },
    }

    joblib.dump(model, out_dir / "logreg_model.pkl")
    joblib.dump(scaler, out_dir / "scaler.pkl")
    joblib.dump(X_encoded.columns.tolist(), out_dir / "feature_columns.pkl")
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    info = train()
    m = info["metrics"]
    print(f"Model accuracy on test set: {m['accuracy']:.4f}")
    print(f"Precision {m['precision']:.4f} | Recall {m['recall']:.4f} | F1 {m['f1']:.4f}")
    print("Model, scaler, feature columns and metadata saved successfully.")
