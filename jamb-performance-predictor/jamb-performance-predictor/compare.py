"""Compare the four models from the thesis (Table 4.1).

As in the thesis appendix, the target here is the raw JAMB score and the models are
scored with regression metrics (RMSE, MAE, R2) on a held-out 20% test split.

Run:  python compare.py
Writes: results/model_comparison.csv
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from train import TARGET, load_dataset

warnings.filterwarnings("ignore")


def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2 = r2_score(true, predicted)
    return mae, rmse, r2


def main():
    df = load_dataset()
    print(df.head(7))

    y = df[TARGET]
    X = df.drop(columns=[TARGET])  # thesis keeps every other column here, Student_ID included

    num_features = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_features = [c for c in X.columns if c not in num_features]
    preprocessor = ColumnTransformer([
        ("OneHotEncoder", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ("StandardScaler", StandardScaler(), num_features),
    ])
    X = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(),
        "K-Nearest Neighbors": KNeighborsRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Support Vector Machine": SVC(),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        tr_mae, tr_rmse, tr_r2 = evaluate_model(y_train, model.predict(X_train))
        te_mae, te_rmse, te_r2 = evaluate_model(y_test, model.predict(X_test))

        print(name)
        print("Model performance for Training set")
        print(f"- Root Mean Squared Error: {tr_rmse:.4f}")
        print(f"- Mean Absolute Error: {tr_mae:.4f}")
        print(f"- R2 Score: {tr_r2:.4f}")
        print("-" * 34)
        print("Model performance for Test set")
        print(f"- Root Mean Squared Error: {te_rmse:.4f}")
        print(f"- Mean Absolute Error: {te_mae:.4f}")
        print(f"- R2 Score: {te_r2:.4f}")
        print("=" * 35 + "\n")
        rows.append({"Model": name, "RMSE": te_rmse, "MAE": te_mae, "R2": te_r2})

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/model_comparison.csv", index=False)
    print("Saved results/model_comparison.csv")


if __name__ == "__main__":
    main()
