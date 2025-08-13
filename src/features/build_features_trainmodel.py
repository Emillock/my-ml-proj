import json
import re

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def to_df_func(X):
    return pd.DataFrame(X, columns=numeric_cols + categorical_cols + binary_cols + monthly_cols)


def str_to_int_func(X):
    return X.apply(pd.to_numeric, errors="coerce")


def winsorize_array(X, limits=(0.01, 0.01)):
    X = np.asarray(X, dtype=float)
    out = np.empty_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        out[:, j] = winsorize(col, limits=limits).data
    return out


def val_pattern():
    arr = []
    for second in range(1, 7):
        for first in range(2, 22):
            s = f"val{first}_{second}"
            if s in ("val3_6", "val3_5", "val3_4"):
                continue
            arr.append(s)
    return arr


numeric_cols = ["age", "tenure", "age_dev", "dev_num"]

binary_cols = ["is_dualsim", "is_featurephone", "is_smartphone"]

categorical_cols = ["trf", "gndr", "dev_man", "device_os_name", "simcard_type", "region"]

monthly_cols = val_pattern()


class ComputeFeatureWeights(BaseEstimator, TransformerMixin):
    def __init__(self, default_weight=6, pattern=r"^val(\d+)_(\d+)$"):
        self.default_weight = default_weight
        self.pattern = re.compile(pattern)

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            cols = list(X.columns)
        else:
            X = pd.DataFrame(X)
            cols = list(X.columns)

        weights = []
        for c in cols:
            m = self.pattern.match(str(c))
            if m:
                try:
                    suffix = int(m.group(2))
                except Exception:
                    weights.append(self.default_weight)
                    continue
                if 1 <= suffix <= 6:
                    w = 7 - suffix
                else:
                    w = self.default_weight
            else:
                w = self.default_weight
            weights.append(float(w))

        self.feature_names_in_ = cols
        self.feature_weights_ = np.asarray(weights, dtype=float)
        return self

    def transform(self, X):
        return X


def main():
    df = pd.read_parquet("./data/external/multisim_dataset.parquet")

    target = "target"
    X = df.drop(columns=[target])
    y = df[[target]]

    str_to_int = FunctionTransformer(func=str_to_int_func, validate=False)

    winsor_transformer = FunctionTransformer(func=winsorize_array)

    to_df = FunctionTransformer(func=to_df_func, validate=False)

    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("str_to_int", str_to_int),
                        ("impute", SimpleImputer(strategy="median")),
                        ("winsorize", winsor_transformer),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encode", CatBoostEncoder()),
                    ]
                ),
                categorical_cols,
            ),
            (
                "bin",
                Pipeline(
                    [
                        ("str_to_int", str_to_int),
                        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                    ]
                ),
                binary_cols,
            ),
            ("monthly", "passthrough", monthly_cols),
        ]
    )
    weights_step_name = "compute_weights"

    preproc_pipeline = Pipeline(
        [
            (weights_step_name, ComputeFeatureWeights(default_weight=6)),
            ("preproc", preprocessor),
            ("to_df", to_df),
        ]
    )

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X_transformed = preproc_pipeline.fit_transform(X, y)

    # joblib.dump(preproc_pipeline, './data/interim/preproc_pipeline.joblib', compress=3)
    weights = preproc_pipeline.named_steps[weights_step_name].feature_weights_.tolist()
    with open("./data/interim/multisim_weights.json", "w") as f:
        json.dump(weights, f)
    df_full = X_transformed.join(y)
    df_full.to_parquet("./data/processed/multisim_dataset.parquet")


if __name__ == "__main__":
    main()
