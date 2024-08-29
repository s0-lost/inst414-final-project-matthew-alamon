# inflation.py
import pandas as pd

# Function to load the transformed Food CPI data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to create a mapping to represent months with values.
# The mapping represents months with values in order for inflation calculations to work as intended.
# Also included is an inverse mapping from numeric values to month names for later use.
def map_months(df):
    month_mapping = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    month_inverse_mapping = {v: k for k, v in month_mapping.items()}
    df['Period'] = df['Period'].map(month_mapping)
    return df, month_inverse_mapping

# Function to ensure data is sorted and clean
def clean_data(df):
    df = df.dropna(subset=['Period'])
    df = df.sort_values(by=['Category', 'Year', 'Period'])
    return df

# Function to calculate the inflation rate
def calculate_inflation(df, month_inverse_mapping):
    inflation_data = []
    for category in df['Category'].unique():
        category_df = df[df['Category'] == category]
        for i in range(1, len(category_df)):
            prev_row = category_df.iloc[i - 1]
            curr_row = category_df.iloc[i]
            
            prev_cpi = prev_row['CPI']
            curr_cpi = curr_row['CPI']
            
            if (curr_row['Year'] == prev_row['Year'] and curr_row['Period'] == prev_row['Period'] + 1) or \
               (curr_row['Year'] == str(int(prev_row['Year']) + 1) and curr_row['Period'] == 1 and prev_row['Period'] == 12):
                if prev_cpi != 0:
                    inflation_rate = ((curr_cpi - prev_cpi) / prev_cpi) * 100
                else:
                    inflation_rate = None
                
                curr_period = f"{month_inverse_mapping[curr_row['Period']]} {curr_row['Year']}"
                
                inflation_data.append({
                    'Period': curr_period,
                    'Category': category,
                    'Inflation Rate': inflation_rate
                })
    inflation_df = pd.DataFrame(inflation_data)
    return inflation_df

# Function to save the inflation data to CSV
def save_data(df, file_path):
    df.to_csv(file_path, index=False)
    print(f"Inflation data saved to: {file_path}")