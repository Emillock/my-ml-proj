import pickle
import re
import os
import json

import joblib
import numpy as np
import pandas as pd
from category_encoders import CatBoostEncoder
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


def main():
    df = pd.read_parquet('./data/processed/multisim_dataset.parquet')

    target = 'target'
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    path = "./data/interim/multisim_weights.json"   # change to your file
    with open(path, "r", encoding="utf-8") as file:
        weights = json.load(file)
    params = {'n_estimators': 186, 'max_depth': 10, 'learning_rate': 0.01155364929483116, 'subsample': 0.5596769098694525, 'colsample_bytree': 0.9364342412798315, 'min_child_weight': 7}
    xgb = XGBClassifier(**params, feature_weights=weights)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    xgb.fit(X_train, y_train)
    filename = './models/xgb_trainmodel.pkl'

    with open(filename, 'wb') as file:
        pickle.dump(xgb, file)


if __name__ == "__main__":
    main()
