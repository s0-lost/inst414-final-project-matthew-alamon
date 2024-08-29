import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime

# Function to load the inflation data.
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to parse period strings into datetime objects.
def parse_period(period_str, month_mapping):
    try:
        month_name, year = period_str.split(' ')
        month = month_mapping[month_name]
        return datetime(int(year), month, 1)
    except:
        return None

# Function to format datetime objects back into period strings.
def format_period(date, reverse_month_mapping):
    month = reverse_month_mapping[date.month]
    year = date.year
    return f"{month} {year}"

# Function to convert periods to numerical values for modeling purposes.
def convert_periods_to_numeric(df):
    df['Period_num'] = (df['Period'].dt.year - df['Period'].dt.year.min()) * 12 + df['Period'].dt.month
    return df

# Function to fit a linear regression model and predict future inflation rates.
def predict_inflation(category_df, future_dates):
    X = category_df[['Period_num']]
    y = category_df['Inflation Rate']
    model = LinearRegression()
    model.fit(X, y)
    
    # Preparing future periods DataFrame with the same feature name to prevent warnings regarding validity of feature names.
    future_periods = np.array([(date.year - category_df['Period'].dt.year.min()) * 12 + date.month for date in future_dates]).reshape(-1, 1)
    future_periods_df = pd.DataFrame(future_periods, columns=['Period_num'])
    
    future_inflation = model.predict(future_periods_df)
    return future_inflation

# Function to save data to a CSV file.
def save_data(df, file_path):
    df.to_csv(file_path, index=False)

# Function to merge historical and forecasted inflation data.
def merge_data(inflation_df, forecast_df):
    merged_df = pd.concat([
        inflation_df[['Period', 'Category', 'Inflation Rate']], 
        forecast_df.rename(columns={'Predicted Inflation Rate': 'Inflation Rate'})
    ], ignore_index=True)
    merged_df['Period'] = pd.to_datetime(merged_df['Period'], format='%B %Y')
    merged_df = merged_df.sort_values(by=['Category', 'Period'])
    return merged_df

def generate_future_dates(start_date, end_date):
    """Generates a list of monthly future dates from start_date to end_date."""
    return pd.date_range(start=start_date, end=end_date, freq='MS').tolist()
