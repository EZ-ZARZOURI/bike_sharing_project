from data_loader import load_data, aggregate_hourly
from features import create_time_features, create_lag_features
from model import get_baseline_model, get_advanced_model
from evaluate import print_metrics

import pandas as pd


def temporal_split(df, split_ratio=0.8):
    split_index = int(len(df) * split_ratio)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test


def main():

    print("Loading data...")
    df = load_data()

    print("Aggregating hourly demand...")
    df = aggregate_hourly(df)

    df = create_time_features(df)
    df = create_lag_features(df)

    df = df.dropna()

    features = [
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "lag_1h",
        "lag_24h",
        "rolling_24h"
    ]

    target = "demand"

    train, test = temporal_split(df)

    X_train = train[features]
    y_train = train[target]

    X_test = test[features]
    y_test = test[target]

    # Baseline
    baseline_model = get_baseline_model()
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)

    print_metrics("Baseline Linear Regression", y_test, y_pred_baseline)

    # Advanced
    advanced_model = get_advanced_model()
    advanced_model.fit(X_train, y_train)
    y_pred_advanced = advanced_model.predict(X_test)

    print_metrics("Random Forest", y_test, y_pred_advanced)


if __name__ == "__main__":
    main()
