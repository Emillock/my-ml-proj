import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier



def main():
    df = pd.read_parquet("./data/processed/ramen-ratings.parquet")

    target = "Stars"

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    filename = "./models/rf_fe2.pkl"

    with open(filename, "rb") as f:
        rf: XGBClassifier = pickle.load(f)

    y_pred = rf.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"R² Score: {r2:.3f}")


if __name__ == "__main__":
    main()
