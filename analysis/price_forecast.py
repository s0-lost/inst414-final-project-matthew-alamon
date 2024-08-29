#price_forecast.py
import pandas as pd

# Defining the directory to store the processed data.
data_dir = 'data/'

def load_data():
    """Load the transformed supermarket sales and predictive inflation data."""
    sales_file_path = f'{data_dir}transformed_supermarket_sales.csv'
    sales_df = pd.read_csv(sales_file_path)

    inflation_file_path = f'{data_dir}predictive_inflation.csv'
    inflation_df = pd.read_csv(inflation_file_path)
    
    return sales_df, inflation_df

def calculate_averages(sales_df):
    """Calculate the average total and average unit price from 2019 data."""
    average_total_2019 = sales_df['Total'].mean()
    average_unit_price_2019 = sales_df['Unit price'].mean()

    return average_total_2019, average_unit_price_2019

def apply_inflation(value, inflation_rate):
    """Apply inflation or deflation to a value based on the given inflation rate."""
    return value * (1 + inflation_rate / 100)

def adjust_price(year, inflation_df, average_total_2019, average_unit_price_2019):
    """Adjust price based on historical inflation data."""
    start_year = 2014
    end_year = 2030

    if year < start_year or year > end_year:
        raise ValueError(f"Year must be between {start_year} and {end_year}.")
    
    # Initialize updated values
    updated_total = average_total_2019
    updated_unit_price = average_unit_price_2019
    
    # Adjust prices based on historical inflation data
    for index, row in inflation_df.iterrows():
        period = row['Period']
        inflation_rate = row['Inflation Rate']
        
        month_name, inflation_year = period.split(' ')
        inflation_year = int(inflation_year)
        
        if inflation_year < 2019:
            # Applying deflation model for years before 2019
            if inflation_year <= year:
                updated_total /= (1 + inflation_rate / 100)
                updated_unit_price /= (1 + inflation_rate / 100)
        else:
            # Applying inflation model for years 2019 and after
            if inflation_year <= year:
                updated_total *= (1 + inflation_rate / 100)
                updated_unit_price *= (1 + inflation_rate / 100)
    
    return updated_total, updated_unit_price