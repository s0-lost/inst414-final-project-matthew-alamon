import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime
import numpy as np

def load_data(file_path):
    """
    Function to load CSV data into a DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(data):
    """
    Function to ensure that 'Period' is a datetime column and set it as the index.
    """
    data['Period'] = pd.to_datetime(data['Period'], format='%B %Y')
    data.set_index('Period', inplace=True)
    return data

def fit_arima_model(data, category):
    """
    Function to fit an ARIMA model to the category data.
    """
    try:
        # Filters data for the specific category.
        category_data = data[data['Category'] == category]['Inflation Rate']
        
        # Ensures the data is not empty.
        if category_data.empty:
            raise ValueError(f"No data available for category: {category}")
        
        # Fits the ARIMA model.
        model = ARIMA(category_data, order=(5,1,0))  # Adjust order as needed
        model_fit = model.fit()
        
        return model_fit
    except Exception as e:
        print(f"Error fitting ARIMA model for category {category}: {e}")
        return None

def forecast_inflation(model_fit, steps):
    """
    Function to forecast future values using the fitted ARIMA model.
    """
    if model_fit:
        # Forecasts future values.
        forecast = model_fit.forecast(steps=steps)
        return forecast
    else:
        return np.nan * np.zeros(steps)  # Return NaN values if model fitting failed.

def save_forecast_to_csv(forecasts, file_path):
    """
    Function to save the forecasted data to a CSV file.
    """
    df_forecasts = pd.DataFrame(forecasts)
    df_forecasts.to_csv(file_path, index=False)

def merge_and_save_data(historical_file, forecasted_file, output_file):
    """
    Function to merge historical and forecasted data, sort by category, and save to CSV.
    """
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
