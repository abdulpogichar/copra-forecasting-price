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
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from flask import request
api = Api('645bbb75279f43629698ed4a2f29a51b')
WEATHERBIT_API_KEY = '645bbb75279f43629698ed4a2f29a51b'

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

# Update this line with the correct path
combined_model_path = 'models/combined_model.pkl'

# Load the combined model
loaded_combined_model = joblib.load(combined_model_path)


# Check if the models are already fitted or not
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

from flask import request

@app.route('/save-profile', methods=['POST'])
def save_profile():
    try:
        # Get form data
        email = request.form.get('email')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        address = request.form.get('address')
        city = request.form.get('city')
        country = request.form.get('country')
        postal_code = request.form.get('postal_code')
        about_me = request.form.get('about_me')

        # Update the user's information in the database (assuming you have a User model)
        user = User.query.filter_by(email=email).first()
        if user:
            user.first_name = first_name
            user.last_name = last_name
            user.address = address
            user.city = city
            user.country = country
            user.postal_code = postal_code
            user.about_me = about_me

            db.session.commit()

            return jsonify({'success': True})
        else:
            return jsonify({'error': 'User not found'})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/prediction-result')
def render_prediction_result():
    template_path = 'result.html'
    if not os.path.exists(os.path.join(app.root_path, 'templates', template_path)):
        abort(404)  # Return a 404 error if the template doesn't exist
    return render_template(template_path)

# Other routes...

# ... (your existing imports)

# Other routes...

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    global model_fitted, feature_columnsv

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

        # Check if the models are already fitted
        if not model_fitted:
            model_fitted = True

        # Perform prediction using the loaded/fitted model
        prediction_classification = loaded_combined_model['classification_model'].predict(X)
        prediction_regression = loaded_combined_model['regression_model'].predict(X)

        # Call Weatherbit API to get forecasted weather information
        weather_data = get_weather_data(date)

        # Extract the predicted price and harvest season from your models
        predicted_price = prediction_regression[0]  # Adjust this according to your model output
        predicted_harvest_season = prediction_classification[0]  # Adjust this according to your model output

        response = {
            'prediction_classification': prediction_classification.tolist(),
            'prediction_regression': prediction_regression.tolist(),
            'weather': weather_data,
            'predicted_price': predicted_price,
            'predicted_harvest_season': predicted_harvest_season
        }

        # Render the prediction result template with the prediction and weather information
        return render_template('home/result.html',
                               prediction_classification=response['prediction_classification'],
                               prediction_regression=response['prediction_regression'],
                               weather=response['weather'],
                               predicted_price=response['predicted_price'],
                               predicted_harvest_season=response['predicted_harvest_season'])

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})

# ... (the rest of your Flask app code)


if __name__ == "__main__":
    app.run()
