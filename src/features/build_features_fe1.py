import re

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, RobustScaler
import pyarrow.parquet as pq


class AddCols(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['data_compl_usg_change_m2_m3'] = (
            X['data_compl_usg_local_m2'] - X['data_compl_usg_local_m3']
        ) / X['data_compl_usg_local_m3'].replace(0, 1)

        X['avg_data_compl_usg_local_3m'] = (
            X[['data_compl_usg_local_m2', 'data_compl_usg_local_m3',
                'data_compl_usg_local_m4']]
            .mean(axis=1)
        )

        usage = X[['data_compl_usg_local_m2',
                   'data_compl_usg_local_m3', 'data_compl_usg_local_m4']]
        std_3m = usage.std(axis=1) / usage.mean(axis=1).replace(0, np.nan)
        X['data_compl_usg_std_3m'] = std_3m.replace(np.nan, 0)

        inact_cols = ['tot_inact_status_days_l1m_m2',
                      'tot_inact_status_days_l1m_m3',
                      'tot_inact_status_days_l1m_m4',
                      'tot_inact_status_days_l1m_m5']
        X['inact_momentum'] = X[inact_cols].sum(axis=1) / (30*4)

        return X


def main():
    local_file_path = './data/external/data_usage_production.parquet'
    table = pq.read_table(local_file_path)  # lazy
    df = table.slice(0, 500000).to_pandas()
    df.set_index("telephone_number", inplace=True)

    target = 'data_compl_usg_local_m1'
    X = df.drop(columns=[target])
    y = df[target]

    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=['object', 'category']).columns.tolist()

    to_df = FunctionTransformer(
        func=lambda X: pd.DataFrame(
            X, columns=numeric_cols + categorical_cols),
        validate=False
    )

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('yeojohnson', PowerTransformer(method='yeo-johnson')),
            ('scale', RobustScaler())
        ]), numeric_cols),

        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('encode', CatBoostEncoder())
        ]), categorical_cols),
    ])

    preproc_pipeline = Pipeline([
        ('preproc', preprocessor),
        ('to_df', to_df),
        ('addCols', AddCols()),
    ])

    X_transformed = preproc_pipeline.fit_transform(X, y)
    X_transformed.to_parquet('./data/processed/data_usage_production.parquet')


if __name__ == "__main__":
    main()
