import pandas as pd

def create_time_features(df):
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


def create_lag_features(df):
    df["lag_1h"] = df["demand"].shift(1)
    df["lag_24h"] = df["demand"].shift(24)
    df["rolling_24h"] = df["demand"].rolling(24).mean()

    return df
