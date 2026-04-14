import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from loan_model import FeatureEngineering


def main():
    root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(root, "LoanGuard_data.csv")
    out_dir = os.path.join(root, "artifacts")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(data_path)

    # Match notebook: drop Applicant_ID
    if "Applicant_ID" in df.columns:
        df = df.drop(columns=["Applicant_ID"])

    # Target
    if "Loan_Approved" not in df.columns:
        raise ValueError("Expected target column 'Loan_Approved' in dataset.")

    # Notebook imputes categorical columns with most_frequent; mirror that here for target too
    # (keeps training deterministic and avoids dropping rows).
    y_series = df["Loan_Approved"]
    if y_series.isna().any():
        mode = y_series.mode(dropna=True)
        if mode.empty:
            raise ValueError("Loan_Approved column is entirely missing.")
        df["Loan_Approved"] = y_series.fillna(mode.iloc[0])

    y_raw = df["Loan_Approved"].astype(str)
    y = y_raw.map({"No": 0, "Yes": 1})
    if y.isna().any():
        bad = sorted(y_raw[y.isna()].unique().tolist())
        raise ValueError(f"Unexpected Loan_Approved labels: {bad}")

    X = df.drop(columns=["Loan_Approved"])

    categorical_cols = [
        "Employment_Status",
        "Marital_Status",
        "Loan_Purpose",
        "Property_Area",
        "Gender",
        "Employer_Category",
    ]
    # Notebook label-encoded Education_Level
    ordinal_cols = ["Education_Level"]

    numeric_cols = [c for c in X.columns if c not in (categorical_cols + ordinal_cols)]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )
    ordinal_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Stable encoding, similar intent to LabelEncoder for a single column
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
            ("ord", ordinal_pipe, ordinal_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    model = Pipeline(
        steps=[
            ("preprocess", pre),
            ("fe", FeatureEngineering()),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )

    model.fit(X, y)

    joblib.dump(model, os.path.join(out_dir, "loan_approval_model.joblib"))

    meta = {
        "target_mapping": {"No": 0, "Yes": 1},
        "input_columns": X.columns.tolist(),
        "categorical_cols": categorical_cols,
        "ordinal_cols": ordinal_cols,
        "numeric_cols": numeric_cols,
    }
    joblib.dump(meta, os.path.join(out_dir, "model_meta.joblib"))

    print("Saved:")
    print(f"- {os.path.join(out_dir, 'loan_approval_model.joblib')}")
    print(f"- {os.path.join(out_dir, 'model_meta.joblib')}")


if __name__ == "__main__":
    main()

