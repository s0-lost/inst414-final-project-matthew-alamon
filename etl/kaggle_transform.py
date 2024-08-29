# kaggle_transform.py
import pandas as pd

# Function to load the data from the CSV file
def load_data(input_file):
    df = pd.read_csv(input_file)
    return df

# Function to filter the dataset to only include entries with 'Food and beverages' in the Product line for better application in this project.
def filter_data(df):
    filtered_df = df[df['Product line'] == 'Food and beverages']
    return filtered_df

# Function to remove the specified columns, which are unneccessary demographic information regarding each data entry 
# (not needed for the scope of this project.)
def remove_columns(df):
    columns_to_remove = ['Invoice ID', 'Branch', 'City', 'Customer type', 'Gender', 'Payment', 'Rating']
    transformed_df = df.drop(columns=columns_to_remove)
    return transformed_df

# Function to save the transformed data to a new CSV file
def save_data(df, output_file):
    df.to_csv(output_file, index=False)
    print(f"Data saved to: {output_file}")