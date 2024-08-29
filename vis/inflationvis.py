import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Define the directory to store the processed data.
data_dir = 'data/'

def load_data(file_path):
    """Load CSV data into a DataFrame."""
    return pd.read_csv(file_path)

def plot_inflation_data(df):
    """Plot predictive inflation data by category and save to a .png file."""
    # Print the unique categories to verify data.
    print("Available categories in the data:")
    print(df['Category'].unique())

    # Define the categories to plot.
    categories = ['Food and Beverage', 'Food Away from home', 'Food at home']

    # Initialize the figure.
    fig = go.Figure()

    # Plot the data for each category.
    for category in categories:
        # Filter the data for the current category.
        category_df = df[df['Category'] == category].copy()
        
        # Check if there is data for this category.
        if category_df.empty:
            print(f"No data found for category: {category}")
            continue
        
        # Convert 'Period' to datetime format to extract the year.
        category_df['Period'] = pd.to_datetime(category_df['Period'], format='%B %Y')
        category_df.sort_values('Period', inplace=True)  # Sort by date
        
        # Add a trace for the category to synthesize the line graph.
        fig.add_trace(go.Scatter(
            x=category_df['Period'],
            y=category_df['Inflation Rate'],
            mode='lines+markers',
            name=category,
            hoverinfo='text',
            text=category_df.apply(lambda row: f"{row['Period'].strftime('%B %Y')}: {row['Inflation Rate']:.2f}%", axis=1)
        ))

    # Update the layout for the figure.
    fig.update_layout(
        title='Predictive Inflation Rate by Category',
        xaxis_title='Year',
        yaxis_title='Inflation Rate (%)',
        xaxis=dict(
            tickformat='%Y',
            rangeslider=dict(visible=True)
        ),
        hovermode='x unified'
    )

    # Save the plot to a .png file.
    output_file = 'predictive_inflation_rate_by_category.png'
    pio.write_image(fig, output_file)
    print(f"Plot saved to {output_file}")

    fig.show()
