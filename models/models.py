# custom_models.py

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class TimeSeriesRandomForestRegressor(RandomForestRegressor):
    def fit(self, X, y):
        # Extract feature names
        self.feature_names_ = X.columns.tolist()

        # Ensure data is sorted by date before fitting
        X_sorted = X.sort_values(by=self.feature_names_)
        y_sorted = y[X_sorted.index]

        super().fit(X_sorted, y_sorted)

    def predict(self, X):
        try:
            # Ensure the order of features is correct
            X_ordered = X[self.feature_names_]

            # Perform prediction using the loaded/fitted model
            predicted = super().predict(X_ordered)
            return predicted

        except ValueError as ve:
            print("Error during prediction:", ve)
            print("Provided Feature Columns:", X_ordered.columns.tolist())
            raise ve
