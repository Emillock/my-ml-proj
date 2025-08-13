import pickle

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier


def main():
    df = pd.read_parquet("./data/processed/multisim_dataset.parquet")

    target = "target"

    X = df.drop(columns=[target])
    y = df[target]

    filename = "./models/xgb_trainmodel.pkl"

    with open(filename, "rb") as f:
        xgb: XGBClassifier = pickle.load(f)

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    print("Precision:", cross_val_score(xgb, X, y, cv=cv, scoring="precision").mean())
    print("Recall:", cross_val_score(xgb, X, y, cv=cv, scoring="recall").mean())
    print("F1 Score:", cross_val_score(xgb, X, y, cv=cv, scoring="f1").mean())


if __name__ == "__main__":
    main()
