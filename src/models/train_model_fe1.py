import pickle
import re
import os

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


def main():
    df = pd.read_parquet('./data/processed/data_usage_production.parquet')

    target = 'data_compl_usg_local_m1'
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        n_jobs=-1, 
    )
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    rf.fit(X_train, y_train)
    filename = './models/rf_fe1.pkl'
    # os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as file:
        pickle.dump(rf, file)


if __name__ == "__main__":
    main()
