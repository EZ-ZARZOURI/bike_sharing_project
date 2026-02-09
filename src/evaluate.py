import numpy as np


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )


def print_metrics(name, y_true, y_pred):
    print(f"\nModel: {name}")
    print(f"MAE: {mae(y_true, y_pred):.2f}")
    print(f"sMAPE: {smape(y_true, y_pred):.2f}%")
