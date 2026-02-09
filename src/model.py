from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def get_baseline_model():
    return LinearRegression()


def get_advanced_model():
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
