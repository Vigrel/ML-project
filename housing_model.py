import os

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

HOUSING_PATH = os.path.join("datasets", "housing")


def load_housing_data(housing_path=HOUSING_PATH) -> pd.DataFrame:
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def get_model() -> LinearRegression:
    housing = load_housing_data().loc[:, ["housing_median_age", "median_house_value"]]
    X_train, _, y_train, _ = train_test_split(
        housing.iloc[:, [0]], housing.iloc[:, [1]], test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


MODEL = get_model()
