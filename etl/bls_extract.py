# bls_extract.py
import os
import requests
import json
import pandas as pd
from datetime import datetime

# Function to set up the data directory
def setup_data_directory(data_dir='data/'):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    return data_dir

# Function to fetch data from BLS API
def fetch_bls_data(series_ids, start_year, end_year, start_month, end_month):
    # API URL for BLS Public Data API 1.0; this is a public API with no key required, but can be limited to only up to 25 calls a day.
    api_url = 'https://api.bls.gov/publicAPI/v1/timeseries/data/'
    
    # Series IDs for Food CPI. This was found online from this link: https://www.bls.gov/cpi/tables/relative-importance/weight-update-comparison-2023.htm
    series_ids = ['CUUR0000SAF', 'CUUR0000SAF11', 'CUUR0000SEFV']

    # Gets current date for active updating purposes.
    now = datetime.now()
    current_year = now.year
    current_month = now.month

    # Defines date range for API call. Keep in mind the API is limited to 10 years maximum, meaning that this may throw errors in the future.
    start_year = 2014
    end_year = current_year
    start_month = 1
    end_month = current_month

    # Preparation for the request payload.
    data = {
        "seriesid": series_ids,
        "startyear": str(start_year),
        "endyear": str(end_year),
        "startmonth": f'{start_month:02}',
        "endmonth": f'{end_month:02}'
    }
    headers = {'Content-type': 'application/json'}
    response = requests.post(api_url, data=json.dumps(data), headers=headers)
    return response

# Function to process the response from BLS API
def process_response(response):
    # Check response status and process data if the response is successful.
    if response.status_code == 200:
        json_data = response.json()
        if json_data['status'] == 'REQUEST_SUCCEEDED':
            results = json_data['Results']['series']
            all_data = []
            
            for series in results:
                series_id = series['seriesID']
                for item in series['data']:
                    year = item['year']
                    period = item['period']
                    value = item['value']
                    all_data.append([series_id, year, period, value])
                    
            # Convert to DataFrame for saving the file as a CSV.
            return pd.DataFrame(all_data, columns=['Series ID', 'Year', 'Period', 'Value'])
        else:
            print("API request failed:", json_data['message'])
            return None
    else:
        print("Request failed with status code", response.status_code)
        return None

# Function to save the processed data to a CSV file
def save_data(df, data_dir, filename='food_cpi_data.csv'):
    csv_file = os.path.join(data_dir, filename)
    df.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")