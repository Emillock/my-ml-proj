import pickle
import re
import os

import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

class DropUnratedWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def fit(self, X, y):
        # build mask of “good” samples
        mask = (y != "Unrated")
        # filter both X and y
        X_filt = X[mask]
        y_filt = y[mask].astype(float)
        # fit the inner pipeline/regressor
        self.pipeline.fit(X_filt, y_filt)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

def main():
    df = pd.read_parquet('./data/processed/ramen-ratings.parquet')

    target = 'Stars'

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf=RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=0
    )

    model=DropUnratedWrapper(rf)
    model.fit(X_train,y_train)

    filename = './models/rf_fe2.pkl'
    # os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'wb') as file:
        pickle.dump(model, file)


if __name__ == "__main__":
    main()
