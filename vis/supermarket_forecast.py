import pandas as pd
import plotly.graph_objects as go

def get_user_input():
    """
    Function to ask the user for the initial COGS price and the start date.
    """
    print(" ")
    print("The previous visualization presented to you was the combined historical and forecasted inflation data based off of your predictive model. Please check the png, predictive_inflation_rate_by_category.png for the visualization.")
    print("The next visualization will require you to input an initial COGS price and the start date of the COGS price, then create a graph showing the inflation and deflation of your input over time.")
    
    initial_cogs = float(input("Enter the initial COGS price: "))
    
    while True:
        try:
            start_year = int(input("Enter the year of the initial COGS price (between 2014 and 2030): "))
            if 2014 <= start_year <= 2030:
                break
            else:
                print("Invalid input. Please enter a year between 2014 and 2030.")
        except ValueError:
            print("Invalid input. Please enter a valid year.")
    
    while True:
        try:
            start_month = int(input("Enter the month of the initial COGS price (between 1 and 12): "))
            if 1 <= start_month <= 12:
                break
            else:
                print("Invalid input. Please enter a month between 1 and 12.")
        except ValueError:
            print("Invalid input. Please enter a valid month.")
    
    return initial_cogs, start_year, start_month

def load_data():
    """
    Function to load data from CSV files.
    """
    inflation_df = pd.read_csv('data/predictive_inflation.csv')
    sales_df = pd.read_csv('data/transformed_supermarket_sales.csv')
    return inflation_df, sales_df

def calculate_average_gross_margin(sales_df):
    """
    Function to calculate the average gross margin percentage from the supermarket sales data.
    """
    return sales_df['gross margin percentage'].mean()

def apply_inflation_deflation(forecast_df, inflation_df, initial_cogs, start_year, start_month):
    """
    Function to apply inflation or deflation to the projected COGS based on the inflation data.
    """
    forecast_df['Inflation Rate'] = 0
    for index, row in inflation_df.iterrows():
        period = pd.to_datetime(row['Period'], format='%B %Y')
        period_month = period.month
        period_year = period.year
        if period_year in forecast_df['Year'].values:
            mask = (forecast_df['Year'] == period_year) & (forecast_df['Month'] == period_month)
            if period_year < start_year or (period_year == start_year and period_month < start_month):
                forecast_df.loc[mask, 'Inflation Rate'] = -row['Inflation Rate']
            elif period_year > start_year or (period_year == start_year and period_month > start_month):
                forecast_df.loc[mask, 'Inflation Rate'] = row['Inflation Rate']
    
    forecast_df.fillna(0, inplace=True)
    forecast_df['Projected COGS'] = initial_cogs * (1 + forecast_df['Inflation Rate'] / 100).cumprod()
    return forecast_df

def calculate_gross_income(cogs, gross_margin):
    """
    Function to calculate the projected gross income based on the projected COGS and average gross margin percentage.
    """
    return cogs * (gross_margin / 100) / (1 - (gross_margin / 100))

def visualize_data(forecast_df, initial_cogs, start_date, avg_gross_margin):
    """
    Function to visualize the projected COGS and gross income through plotly.
    """
    fig = go.Figure()

    # Adds line for projected COGS, starting from the adjusted COGS.
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Projected COGS'],
        mode='lines',
        name='Projected COGS'
    ))

    # Adds line for projected gross income.
    forecast_df['Projected Gross Income'] = calculate_gross_income(forecast_df['Projected COGS'], avg_gross_margin)
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Projected Gross Income'],
        mode='lines',
        name='Projected Gross Income'
    ))

    # Adds marker for the original COGS.
    fig.add_trace(go.Scatter(
        x=[start_date],
        y=[initial_cogs],
        mode='markers',
        name='Original COGS',
        marker=dict(color='red', size=10)
    ))

    fig.update_layout(title='Projected COGS and Gross Income',
                      xaxis_title='Date',
                      yaxis_title='Amount',
                      legend_title='Legend')
    fig.show()