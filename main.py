import os
import logging
from datetime import datetime
import etl.bls_extract as bls_extract
import etl.bls_transform as bls_transform
import etl.kaggle_transform as kaggle_transform

import pandas as pd
import analysis.inflation as inflation
import analysis.predict as predict
import analysis.predict_ARIMA as predict_ARIMA
import analysis.predict_LinReg as predict_LinReg
import analysis.model_evaluation as model_evaluation
import analysis.price_forecast as price_forecast

import vis.inflationvis as inflationvis
import vis.supermarket_forecast as supermarket_forecast

# Logging configuration.
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    
#-----------------------------------------------------------------------------------#

    # ETL Scripts. Run sequentially.
    # Run bls_extract.py.
    try:
        logger.info("Starting bls_extract.py")
        
        # Set up the data directory
        data_dir = bls_extract.setup_data_directory()
        
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

        # Fetch data from BLS API.
        response = bls_extract.fetch_bls_data(series_ids, start_year, end_year, start_month, end_month)
    
        # Process response and save data if successful
        if response:
            df = bls_extract.process_response(response)
            if df is not None:
                bls_extract.save_data(df, data_dir)

        logger.info("Completed bls_extract.py")
    except Exception as e:
        logger.error(f"Error in bls_extract.py: {e}")
    
    # Run bls_transform.py.
    try:
        logger.info("Starting bls_transform.py")

        # Defining the file paths for the data to be pulled/stored.
        input_file = 'data/food_cpi_data.csv'
        output_file = 'data/transformed_food_cpi_data.csv'
        
        # Loads the CSV file.
        df = bls_transform.load_data(input_file)

        # Rename columns to desired names.
        df = bls_transform.rename_columns(df)

        # Mapping the Series ID to its associated category.
        df = bls_transform.map_category(df)

        # Converting period to month names.
        df = bls_transform.map_period(df)

        # Saves the transformed data to a new CSV file.
        bls_transform.save_data(df, output_file)

        logger.info("Completed bls_transform.py")
    except Exception as e:
        logger.error(f"Error in bls_transform.py: {e}")

    # Run kaggle_transform.py.
    try:
        logger.info("Starting kaggle_transform.py")

        # Defining the file paths for the data to be pulled/stored.
        input_file = 'data/supermarket_sales.csv'
        output_file = 'data/transformed_supermarket_sales.csv'
        
        # Load the data from the CSV file
        df = kaggle_transform.load_data(input_file)
        
        # Filter the data to only include 'Food and beverages' entries
        df = kaggle_transform.filter_data(df)
        
        # Remove unnecessary columns
        df = kaggle_transform.remove_columns(df)
        
        # Save the transformed data to a new CSV file
        kaggle_transform.save_data(df, output_file)
        print("Data saved to: ", output_file)

        logger.info("Completed kaggle_transform.py")
    except Exception as e:
        logger.error(f"Error in kaggle_transform.py: {e}")

#-----------------------------------------------------------------------------------#

    # Analysis Scripts. Runs based off user input.
    # Prompt user to select a predictive model
    print("Select a predictive model to use:")
    print("1. SARIMA: Seasonal Autoregressive Integrated Moving-Average model, most commonly used for time series forecasting. This one is most recommended for this project.")
    print("2. ARIMA: Autoregressive Integrated Moving-Average model, similar to SARIMA but without the seasonal component.")
    print("3. Linear Regression: A simple linear model that assumes a linear relationship between the input and output variables.")

    model_choice = input("Enter the number corresponding to your choice (1, 2, or 3): ")

    while model_choice not in ['1', '2', '3']:
        print("Invalid choice. Please enter 1, 2, or 3.")
        model_choice = input("Enter the number corresponding to your choice (1, 2, or 3): ")

    data_dir = 'data/'
    historical_file_path = f'{data_dir}inflation.csv'
    categories = ['Food at home', 'Food Away from home', 'Food and Beverage']  # List of categories

    if model_choice == '1':
        # Run SARIMA analysis
        try:
            logger.info("Starting SARIMA analysis (predict.py)")

            forecasted_file_path = f'{data_dir}forecasted_inflation.csv'
            output_file_path = f'{data_dir}predictive_inflation.csv'

            data = predict.load_data(historical_file_path)
            data = predict.preprocess_data(data)
            
            all_forecasts = []

            # Determine the number of months to forecast until December 2030
            last_date = data.index[-1]
            end_date = pd.Timestamp('2030-12-01')
            steps = (end_date.year - last_date.year) * 12 + (end_date.month - last_date.month)

            for category in categories:
                print(f"Processing category: {category}")
                model_fit = predict.fit_sarima_model(data, category)
                forecast = predict.forecast_inflation(model_fit, steps)
                
                # Generate future dates for the forecast
                future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, len(forecast) + 1)]
                
                # Prepare the forecasted data for saving
                forecast_data = {
                    'Period': future_dates,
                    'Category': category,
                    'Inflation Rate': forecast
                }
                all_forecasts.append(pd.DataFrame(forecast_data))

            # Save the forecasted data to CSV
            forecasted_df = pd.concat(all_forecasts, ignore_index=True)
            predict.save_forecast_to_csv(forecasted_df, forecasted_file_path)
            print("Forecasted data saved to: ", forecasted_file_path)

            # Merge historical and forecasted data and save to CSV
            predict.merge_and_save_data(historical_file_path, forecasted_file_path, output_file_path)
            print("Predictive data (both historical and forecasted combined) saved to: ", output_file_path)

            logger.info("Completed SARIMA analysis (predict.py)")
        except Exception as e:
            logger.error(f"Error in predict.py: {e}")

    elif model_choice == '2':
        # Run ARIMA prediction analysis
        try:
            logger.info("Starting ARIMA analysis (predict_ARIMA.py)")

            forecasted_file_path = f'{data_dir}forecasted_inflation.csv'
            eval_file_path = f'{data_dir}predictive_inflation_ARIMA.csv'
            output_file_path = f'{data_dir}predictive_inflation.csv'

            data = predict_ARIMA.load_data(historical_file_path)
            data = predict_ARIMA.preprocess_data(data)
            
            all_forecasts = []

            # Determine the number of months to forecast until December 2030
            last_date = data.index[-1]
            end_date = pd.Timestamp('2030-12-01')
            steps = (end_date.year - last_date.year) * 12 + (end_date.month - last_date.month)

            for category in categories:
                print(f"Processing category: {category}")
                model_fit = predict_ARIMA.fit_arima_model(data, category)
                forecast = predict_ARIMA.forecast_inflation(model_fit, steps)
                
                # Generate future dates for the forecast
                future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, len(forecast) + 1)]
                
                # Prepare the forecasted data for saving
                forecast_data = {
                    'Period': future_dates,
                    'Category': category,
                    'Inflation Rate': forecast
                }
                all_forecasts.append(pd.DataFrame(forecast_data))
            
            # Combine all forecasts into a single DataFrame
            all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
            
            # Save combined forecasts to a CSV file
            predict_ARIMA.save_forecast_to_csv(all_forecasts_df, forecasted_file_path)
            print("Forecasted data saved to: ", forecasted_file_path)    

            # Merge historical and forecasted data, sort, and save
            predict_ARIMA.merge_and_save_data(historical_file_path, forecasted_file_path, output_file_path)
            predict_ARIMA.merge_and_save_data(historical_file_path, forecasted_file_path, eval_file_path)   
            print("Predictive data (both historical and forecasted combined) saved to: ", output_file_path)

            logger.info("Completed ARIMA analysis (predict_ARIMA.py)")
        except Exception as e:
            logger.error(f"Error in predict_ARIMA.py: {e}")

    elif model_choice == '3':
        # Run Linear Regression analysis
        try:
            logger.info("Starting Linear Regression Analysis (predict_LinReg.py)")

            forecasted_file_path = f'{data_dir}forecasted_inflation.csv'
            eval_file_path = f'{data_dir}predictive_inflation_LinReg.csv'
            output_file_path = f'{data_dir}predictive_inflation.csv'

            data = predict_LinReg.load_data(historical_file_path)
            
            # Convert 'Period' column to datetime
            data['Period'] = pd.to_datetime(data['Period'], format='%B %Y')
            
            # Convert periods to numeric values
            data = predict_LinReg.convert_periods_to_numeric(data)
            
            all_forecasts = []

            # Determine the number of months to forecast until December 2030
            last_date = data['Period'].max()
            end_date = pd.Timestamp('2030-12-01')
            future_dates = predict_LinReg.generate_future_dates(last_date, end_date)

            for category in categories:
                print(f"Processing category: {category}")
                category_df = data[data['Category'] == category]
                forecast = predict_LinReg.predict_inflation(category_df, future_dates)
                
                # Prepare the forecasted data for saving
                forecast_data = {
                    'Period': future_dates,
                    'Category': category,
                    'Inflation Rate': forecast
                }
                all_forecasts.append(pd.DataFrame(forecast_data))
            
            # Combine all forecasts into a single DataFrame
            all_forecasts_df = pd.concat(all_forecasts, ignore_index=True)
            
            # Save combined forecasts to a CSV file
            predict_LinReg.save_data(all_forecasts_df, forecasted_file_path)
            print("Forecasted data saved to: ", forecasted_file_path)

            # Merge historical and forecasted data, sort, and save
            merged_data = predict_LinReg.merge_data(data, all_forecasts_df)
            predict_LinReg.save_data(merged_data, output_file_path)
            predict_LinReg.save_data(merged_data, eval_file_path)
            print("Predictive data (both historical and forecasted combined) saved to: ", output_file_path)

            logger.info("Completed Linear Regression Analysis (predict_LinReg.py)")
        except Exception as e:
            logger.error(f"Error in predict_LinReg.py: {e}")

    # Ask the user if they want to run evaluation metrics
    while True:
        run_evaluation = input("Do you want to run evaluation metrics? NOTE: You must run all 3 models before you can properly run evaluation metrics. (Y/N): ").upper()
        if run_evaluation in ['Y', 'N']:
            break
        else:
            print("Invalid input. Please enter 'Y' for Yes or 'N' for No.")

    if run_evaluation == 'Y':
        # Run model_evaluation.py.
        try:
            logger.info("Starting model_evaluation.py")

            data_dir = 'c:/Users/Matthew/Documents/INST414/data/'
            
            historical_file_path = f'{data_dir}inflation.csv'
            forecasted_file_paths = [
                f'{data_dir}predictive_inflation.csv',  # SARIMA
                f'{data_dir}predictive_inflation_LinReg.csv',  # Linear Regression
                f'{data_dir}predictive_inflation_ARIMA.csv'  # ARIMA
            ]
            metrics_output_paths = [
                f'{data_dir}metrics_sarima.csv',
                f'{data_dir}metrics_linreg.csv',
                f'{data_dir}metrics_arima.csv'
            ]
            
            historical_data = model_evaluation.load_data(historical_file_path)['Inflation Rate']
            
            for forecasted_file_path, metrics_output_path in zip(forecasted_file_paths, metrics_output_paths):
                forecasted_data = model_evaluation.load_data(forecasted_file_path)['Inflation Rate']
                
                # Align data lengths
                historical_data_aligned, forecasted_data_aligned = model_evaluation.align_data_lengths(historical_data, forecasted_data)
                
                metrics = model_evaluation.calculate_metrics(historical_data_aligned, forecasted_data_aligned)
                model_evaluation.save_metrics_to_csv(metrics, metrics_output_path)

            logger.info("Completed model_evaluation.py")
        except Exception as e:
            logger.error(f"Error in model_evaluation.py: {e}")
    else:
        print("Skipping evaluation metrics.")

    #Run price_forecast.py.
    try:
        logger.info("Starting price_forecast.py")

        # Load the data
        sales_df, inflation_df = price_forecast.load_data()
        
        # Calculate averages
        average_total_2019, average_unit_price_2019 = price_forecast.calculate_averages(sales_df)

        # Get user input
        print("-----------------------------------")
        print("Price Forecasting: Example of Inflation Adjustment")
        print(f"The average total of food and beverage in 2019 based off of supermarket data was ${average_total_2019:.2f}.")
        print(f"The average unit price of food and beverage in 2019 based off of supermarket data was ${average_unit_price_2019:.2f}.")

        while True:
            try:
                input_year = int(input("Enter a desired year between 2014 and 2030 for inflation forecasting: "))
                if 2014 <= input_year <= 2030:
                    break
                else:
                    print("Invalid input. Please enter a year between 2014 and 2030.")
            except ValueError:
                print("Invalid input. Please enter a valid year.")

        # Adjust prices
        updated_total, updated_unit_price = price_forecast.adjust_price(input_year, inflation_df, average_total_2019, average_unit_price_2019)
        
        # Output results
        print(f"The new average total of food and beverage in {input_year} is: ${updated_total:.2f}")
        print(f"The new average unit price of food and beverage in {input_year} is: ${updated_unit_price:.2f}")

        logger.info("Completed price_forecast.py")
    except Exception as e:
        logger.error(f"Error in price_forecast.py: {e}")

#-----------------------------------------------------------------------------------#

    # Visualization Scripts. Run Sequentially.
    # Run inflationvis.py.
    try:
        logger.info("Starting inflationvis.py")

        data_dir = 'c:/Users/Matthew/Documents/INST414/data/'
        file_path = f'{data_dir}predictive_inflation.csv'
        
        # Load data
        df = inflationvis.load_data(file_path)
        
        # Plot the inflation data
        inflationvis.plot_inflation_data(df)

        logger.info("Completed inflationvis.py")
    except Exception as e:
        logger.error(f"Error in inflationvis.py: {e}")

    # Run supermarket_forecast.py.
    try:
        logger.info("Starting supermarket_forecast.py")

        # Get user input
        initial_cogs, start_year, start_month = supermarket_forecast.get_user_input()
        
        # Load data
        inflation_df, sales_df = supermarket_forecast.load_data()
        
        # Calculate average gross margin
        avg_gross_margin = supermarket_forecast.calculate_average_gross_margin(sales_df)

        # Create a date range for the forecasting period.
        date_range = pd.date_range(start='2014-02-01', end='2030-12-01', freq='MS')
        forecast_df = pd.DataFrame({'Date': date_range})
        forecast_df['Year'] = forecast_df['Date'].dt.year
        forecast_df['Month'] = forecast_df['Date'].dt.month

        # Apply inflation/deflation to the projected COGS
        forecast_df = supermarket_forecast.apply_inflation_deflation(forecast_df, inflation_df, initial_cogs, start_year, start_month)
        start_date = pd.to_datetime(f'{start_year}-{start_month:02d}-01')

        # Visualize the data
        supermarket_forecast.visualize_data(forecast_df, initial_cogs, start_date, avg_gross_margin)

        logger.info("Completed supermarket_forecast.py")
    except Exception as e:
        logger.error(f"Error in supermarket_forecast.py: {e}")


if __name__ == "__main__":
    main()