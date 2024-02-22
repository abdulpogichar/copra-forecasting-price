import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.base import clone

import numpy as np
import joblib

class TimeSeriesRandomForestRegressor(RandomForestRegressor):
    def fit(self, X, y):
        # Assuming X is a DataFrame
        self.feature_names_ = X.columns.tolist()
        super().fit(X, y)

    def predict(self, X):
        try:
            # Ensure the order of features is correct
            X = X[self.feature_names_]
            predicted = super().predict(X)
            return predicted
        except ValueError as ve:
            print("Error during prediction:", ve)
            print("Provided Feature Columns:", X.columns.tolist())
            raise ve
