import os
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def calculate_metrics(historical_data, forecasted_data):
    """Calculate evaluation metrics."""
    mae = mean_absolute_error(historical_data, forecasted_data)
    mse = mean_squared_error(historical_data, forecasted_data)
    rmse = np.sqrt(mse)
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse
    }
    
    return metrics

def save_metrics_to_csv(metrics, file_path):
    """Save the evaluation metrics to a CSV file."""
    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(file_path, index=False)
    print(f"Metrics saved to: {file_path}. Please check them for the evaluation results.")

def align_data_lengths(historical_data, forecasted_data):
    """Align the lengths of historical and forecasted data."""
    min_length = min(len(historical_data), len(forecasted_data))
    return historical_data[:min_length], forecasted_data[:min_length]
