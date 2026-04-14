from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Replicates notebook FE:
    - add DTI_Ratio_sq = DTI_Ratio ** 2
    - add Credit_Score_sq = Credit_Score ** 2
    - drop DTI_Ratio and Credit_Score
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Expect a pandas.DataFrame due to set_output(transform="pandas")
        if not hasattr(X, "copy"):
            raise TypeError("FeatureEngineering expects a pandas DataFrame input.")

        X = X.copy()

        if "DTI_Ratio" not in X.columns or "Credit_Score" not in X.columns:
            missing = [c for c in ["DTI_Ratio", "Credit_Score"] if c not in X.columns]
            raise ValueError(f"Missing required columns for FE: {missing}")

        X["DTI_Ratio_sq"] = X["DTI_Ratio"].astype(float) ** 2
        X["Credit_Score_sq"] = X["Credit_Score"].astype(float) ** 2
        X = X.drop(columns=["Credit_Score", "DTI_Ratio"])
        return X

