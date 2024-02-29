import pandas as pd
import joblib
import os
import requests
from flask_migrate import Migrate
from flask_minify import Minify
from flask import Flask, render_template, request, jsonify, abort
from apps.config import config_dict
from apps import create_app, db
from models.models import TimeSeriesRandomForestRegressor
from weatherbit.api import Api

api = Api('d784a4b36f034f7cb387201196c69cc6')
WEATHERBIT_API_KEY = 'd784a4b36f034f7cb387201196c69cc6'

DEBUG = (os.getenv('DEBUG', 'False') == 'True')

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try:
    # Load the configuration using the default values
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)
Migrate(app, db)

if not DEBUG:
    Minify(app=app, html=True, js=False, cssless=False)

if DEBUG:
    app.logger.info('DEBUG       = ' + str(DEBUG))
    app.logger.info('DBMS        = ' + app_config.SQLALCHEMY_DATABASE_URI)
    app.logger.info('ASSETS_ROOT = ' + app_config.ASSETS_ROOT)

rf_model = TimeSeriesRandomForestRegressor()

# Check if the model is already fitted or not
model_fitted = False
feature_columns = None  # Initialize feature_columns outside the block

# Function to fetch future weather data from Weatherbit API
def get_weather_data(date):
    try:
        lat_zamboanga = 6.9214
        lon_zamboanga = 122.0790

        api_url = f'https://api.weatherbit.io/v2.0/forecast/daily?key={WEATHERBIT_API_KEY}&lat={lat_zamboanga}&lon={lon_zamboanga}&start_date={date}&days=16'

        # Make the API request
        response = requests.get(api_url)
        response.raise_for_status()
        weather_data = response.json()

        return weather_data

    except requests.exceptions.RequestException as e:
        print(f'Error fetching weather data: {e}')
        return None

@app.route('/')
def render_form():
    return render_template('tables.html')

@app.route('/prediction-result')
def render_prediction_result():
    template_path = 'result.html'
    if not os.path.exists(os.path.join(app.root_path, 'templates', template_path)):
        abort(404)  # Return a 404 error if the template doesn't exist
    return render_template(template_path)


@app.route('/get_prediction_data')
def get_prediction_data():

    sample_prediction_data = [
        {"DATE": "2017-01-31", "PRICE": 38},
        {"DATE": "2017-04-30", "PRICE": 40},
        {"DATE": "2017-07-31", "PRICE": 40},
        {"DATE": "2017-10-31", "PRICE": 41},
        {"DATE": "2018-01-31", "PRICE": 40},
        {"DATE": "2018-04-30", "PRICE": 40},
        {"DATE": "2018-07-31", "PRICE": 32},
        {"DATE": "2018-10-31", "PRICE": 29},
        {"DATE": "2019-01-31", "PRICE": 28},
        {"DATE": "2019-04-30", "PRICE": 25},
        {"DATE": "2017-01-31", "PRICE": 24},
        {"DATE": "2017-04-30", "PRICE": 23},
        {"DATE": "2017-07-31", "PRICE": 24},
        {"DATE": "2017-10-31", "PRICE": 27},
        {"DATE": "2018-01-31", "PRICE": 26},
        {"DATE": "2018-04-30", "PRICE": 33.},
        {"DATE": "2018-07-31", "PRICE": 37.},
        {"DATE": "2018-10-31", "PRICE": 34.},
        {"DATE": "2019-01-31", "PRICE": 40},
        {"DATE": "2019-04-30", "PRICE": 41},

    ]

    return jsonify(sample_prediction_data)

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    global model_fitted, feature_columns  # Declare model_fitted and feature_columns as global

    try:
        # Get form data
        data = request.form

        # Extracting individual values
        day = data.get('day')
        month = data.get('month')
        year = data.get('year')


        if day is None or month is None or year is None:
            raise ValueError("Missing required values for day, month, and year")

        # Convert values to integers
        day = int(day)
        month = int(month)
        year = int(year)

        # Convert to a date type
        date_str = f"{year:04d}-{month:02d}-{day:02d}"
        date_format = "%Y-%m-%d"
        date = pd.to_datetime(date_str, format=date_format)

        # Create a feature matrix (a DataFrame) for prediction
        X = pd.DataFrame({'day': [date.day], 'month': [date.month], 'year': [date.year]})

        # Check if the model is already fitted
        if not model_fitted:

            csv_file_path = r'C:\Users\williiam butcher oi\Desktop\LAST\ThesisTrialv1-master\models\processed.csv'
            df = pd.read_csv(csv_file_path)

            # Ensure the correct order and names of columns in X_train
            feature_columns = ['day', 'month', 'year']
            target_column = 'PRICE (PHP per kg)'

            X_train = df[feature_columns]
            y_train = df[target_column]


            rf_model.fit(X_train, y_train)


            model_fitted = True


        print("Training Feature Columns:", feature_columns)
        print("Prediction Feature Columns:", X.columns)

        # Perform prediction using the loaded/fitted model
        prediction = rf_model.predict(X[feature_columns])

        # Call Weatherbit API to get forecasted weather information
        weather_data = get_weather_data(date)

        response = {'prediction': prediction.tolist(), 'weather': weather_data}

        # Render the prediction result template with the prediction and weather information
        return render_template('home/result.html', prediction=response['prediction'], weather=response['weather'])

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run()
