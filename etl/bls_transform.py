# bls_transform.py
import pandas as pd

# Function to load the CSV file
def load_data(input_file):
    df = pd.read_csv(input_file)
    return df

# Function to rename columns
def rename_columns(df):
    df.rename(columns={'Series ID': 'Category', 'Value': 'CPI'}, inplace=True)
    return df

# Function to map category
def map_category(df):
    category_mapping = {
        'CUUR0000SAF': 'Food and Beverage',
        'CUUR0000SAF11': 'Food at home',
        'CUUR0000SEFV': 'Food Away from home'
    }
    df['Category'] = df['Category'].map(category_mapping)
    return df

# Function to map period to month names
def map_period(df):
    period_mapping = {
        'M01': 'January', 'M02': 'February', 'M03': 'March',
        'M04': 'April', 'M05': 'May', 'M06': 'June',
        'M07': 'July', 'M08': 'August', 'M09': 'September',
        'M10': 'October', 'M11': 'November', 'M12': 'December'
    }
    df['Period'] = df['Period'].map(period_mapping)
    return df

# Function to save the transformed data to a CSV file
def save_data(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")
