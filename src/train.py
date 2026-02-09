from data_loader import load_data, aggregate_hourly
from features import create_time_features, create_lag_features
from model import get_baseline_model, get_advanced_model
from evaluate import print_metrics

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import holidays


def temporal_split(df, split_ratio=0.8):
    split_index = int(len(df) * split_ratio)
    train = df.iloc[:split_index]
    test = df.iloc[split_index:]
    return train, test

def robustness_analysis(test, y_pred_advanced):
    print("\n--- Robustness Analysis ---")

    # Reset indices pour que y_pred_advanced corresponde
    test = test.reset_index(drop=True)

    
    # Pics de demande
    mean_demand = test["demand"].mean()
    std_demand = test["demand"].std()
    threshold_peak = mean_demand + 2 * std_demand
    peaks = test[test["demand"] > threshold_peak].reset_index(drop=True)

    if len(peaks) > 0:
        mae_peaks = mean_absolute_error(peaks["demand"], y_pred_advanced[peaks.index])
        print(f"Number of peak hours: {len(peaks)}")
        print(f"MAE on peak hours: {mae_peaks:.2f}")

        # Graphique pics
        plt.figure(figsize=(12,5))
        plt.plot(peaks["datetime"], peaks["demand"], label="Réel")
        plt.plot(peaks["datetime"], y_pred_advanced[peaks.index], label="RF prévision")
        plt.xlabel("Datetime")
        plt.ylabel("Demand")
        plt.title("Prévision vs Réel - Pics de demande")
        plt.legend()
        plt.show()
        plt.pause(0.1)
    else:
        print("No peak hours found in test set.")


    # Jours fériés
   
    us_holidays = holidays.US(years=test["datetime"].dt.year.unique())
    test["is_holiday"] = test["datetime"].dt.date.isin(us_holidays)

    holidays_data = test[test["is_holiday"]].reset_index(drop=True)
    normal_data = test[~test["is_holiday"]].reset_index(drop=True)

    if len(holidays_data) > 0:
        mae_holidays = mean_absolute_error(holidays_data["demand"], y_pred_advanced[holidays_data.index])
        print(f"MAE on holidays: {mae_holidays:.2f}")

        # Graphique jours fériés
        plt.figure(figsize=(12,5))
        plt.plot(holidays_data["datetime"], holidays_data["demand"], label="Réel")
        plt.plot(holidays_data["datetime"], y_pred_advanced[holidays_data.index], label="RF prévision")
        plt.xlabel("Datetime")
        plt.ylabel("Demand")
        plt.title("Prévision vs Réel - Jours fériés")
        plt.legend()
        plt.show()
        plt.pause(0.1)
    else:
        print("No holiday data in the test set to compute MAE or plot.")

    if len(normal_data) > 0:
        mae_normal = mean_absolute_error(normal_data["demand"], y_pred_advanced[normal_data.index])
        print(f"MAE on normal days: {mae_normal:.2f}")



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

    # Graphique général
    plt.figure(figsize=(12,5))
    plt.plot(test["datetime"], y_test, label="Réel")
    plt.plot(test["datetime"], y_pred_advanced, label="RF prévision")
    plt.xlabel("Datetime")
    plt.ylabel("Demand")
    plt.title("Prévision vs Réel - Test Set")
    plt.legend()
    plt.show()
    plt.pause(0.1)

    # Analyse de robustesse
    robustness_analysis(test, y_pred_advanced)

if __name__ == "__main__":
    main()
