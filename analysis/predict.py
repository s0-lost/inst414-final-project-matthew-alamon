import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import numpy as np

def load_data(file_path):
    """Load CSV data into a DataFrame."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Ensure that 'Period' is a datetime column and set it as the index."""
    data['Period'] = pd.to_datetime(data['Period'], format='%B %Y')
    data.set_index('Period', inplace=True)
    return data

def fit_sarima_model(data, category):
    """Fit a SARIMA model to the category data."""
    try:
        # Filter data for the specific category
        category_data = data[data['Category'] == category]['Inflation Rate']
        
        # Ensure the data is not empty
        if category_data.empty:
            raise ValueError(f"No data available for category: {category}")
        
        # Fit the SARIMA model
        model = SARIMAX(category_data, order=(1,1,1), seasonal_order=(1,1,1,12))  # Adjust orders as needed
        model_fit = model.fit(disp=False)
        
        return model_fit
    except Exception as e:
        print(f"Error fitting SARIMA model for category {category}: {e}")
        return None

def forecast_inflation(model_fit, steps):
    """Forecast future values using the fitted SARIMA model."""
    if model_fit:
        # Forecast future values
        forecast = model_fit.get_forecast(steps=steps).predicted_mean
        return forecast
    else:
        return np.nan * np.zeros(steps)  # Return NaN values if model fitting failed

def save_forecast_to_csv(forecasts, file_path):
    """Save the forecasted data to a CSV file."""
    df_forecasts = pd.DataFrame(forecasts)
    df_forecasts.to_csv(file_path, index=False)

def merge_and_save_data(historical_file, forecasted_file, output_file):
    """Merge historical and forecasted data, sort by category, and save to CSV."""
    historical_data = pd.read_csv(historical_file)
    forecasted_data = pd.read_csv(forecasted_file)

    # Convert 'Period' in historical data to datetime
    historical_data['Period'] = pd.to_datetime(historical_data['Period'], format='%B %Y')

    # Convert 'Period' in forecasted data to datetime
    forecasted_data['Period'] = pd.to_datetime(forecasted_data['Period'])

    # Combine historical and forecasted data
    combined_data = pd.concat([historical_data, forecasted_data], ignore_index=True)

    # Sort by category and then by period
    combined_data.sort_values(by=['Category', 'Period'], inplace=True)

    # Convert 'Period' back to 'Month Year' format
    combined_data['Period'] = combined_data['Period'].dt.strftime('%B %Y')

    # Save the combined data to a new CSV file
    combined_data.to_csv(output_file, index=False)

