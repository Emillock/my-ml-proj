import pandas as pd
import numpy as np
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats.mstats import winsorize
from category_encoders import CatBoostEncoder

def val_pattern():
    arr=[]
    for second in range(1, 7):
        for first in range(2, 22):
            s=f"val{first}_{second}"
            if s in ('val3_6', 'val3_5', 'val3_4'):
                continue
            arr.append(s)
    return arr

numeric_cols=[
 'age',
 'tenure',
 'age_dev',
 'dev_num']

binary_cols=[
 'is_dualsim',
 'is_featurephone',
 'is_smartphone']

categorical_cols=['trf',
 'gndr',
 'dev_man',
 'device_os_name',
 'simcard_type',
 'region']

monthly_cols=val_pattern()

class ComputeFeatureWeights(BaseEstimator, TransformerMixin):
    def __init__(self, default_weight=6, pattern=r'^val(\d+)_(\d+)$'):
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
    df = pd.read_parquet('../../data/external/multisim_dataset.parquet')
    X_train, y_train, X_test, y_test = train_test_split(df.drop(columns=['target']), df['target'], test_size=0.2, random_state=42)
    
    str_to_int = FunctionTransformer(
        func=lambda X: X.apply(pd.to_numeric, errors='coerce'),
        validate=False
    )

    def winsorize_array(X, limits=(0.01, 0.01)):
        X = np.asarray(X, dtype=float)
        out = np.empty_like(X)
        for j in range(X.shape[1]):
            col = X[:, j]
            out[:, j] = winsorize(col, limits=limits).data
        return out

    winsor_transformer = FunctionTransformer(lambda X: winsorize_array(X, limits=(0.01, 0.01)))

    to_df = FunctionTransformer(
        func=lambda X: pd.DataFrame(X, columns=numeric_cols + categorical_cols+binary_cols + monthly_cols),
        validate=False
    )

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('str_to_int', str_to_int),
            ('impute', SimpleImputer(strategy='median')),
            ('winsorize', winsor_transformer)
        ]), numeric_cols),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('encode', CatBoostEncoder())
        ]), categorical_cols),
        ('bin', Pipeline([
            ('str_to_int', str_to_int),
            ('impute', SimpleImputer(strategy='constant', fill_value=0))
        ]),binary_cols),
        ('monthly','passthrough',monthly_cols)
    ])

    preproc_pipeline = Pipeline([
        ('compute_weights', ComputeFeatureWeights(default_weight=6)),
        ('preproc', preprocessor),
        ('to_df', to_df)
    ])
    
    X_train_transformed = preproc_pipeline.fit_transform(X_train, y_train)
    X_train_transformed.to_parquet('../../data/processed/X_train_transformed.parquet')


if __name__ == "__main__":
    main()
