# inst414-final-project-matthew-alamon

# Project Overview
Inflation is an ever-growing channel of concern in the modern society, especially when it comes to food and groceries. This project aims to analyze and visualize the impact inflation has on the food industry, utilizing historical CPI metrics grabbed directly from Public Open and sample supermarket sales data from 2019. This project also applies predictive models to inflation data to further predict the cost of food-related goods in future years and visualizes these for a deeper understanding of how inflation is directly impacting the food industry.

# Datasets Used
- Data extracted from BLS Public Data API 1.0.
- Kaggle dataset on Supermarket Sales: https://www.kaggle.com/datasets/aungpyaeap/supermarket-sales

# Techniques Employed
- Data Extraction: Using the Bureau of Labor Statistics API to obtain CPI data.
- Data Transformation: Cleaning and processing raw data from the BLS API as well as the Kaggle Supermarket Sales dataset to derive meaningful insights.
- Predictive Modeling: Utilizing a time series analysis model (SARIMA) to represent forecasted inflation.
- Visualization: Creating interactive and static visualizations to represent inflation trends across different categories.

# Setup Instructions
1. Clone the Repository

   - git clone https://github.com/yourusername/yourrepository.git
   cd yourrepository

2. Set Up and Activate the Virtual Environment using the following code:
   ```python -m venv .venv```

   Then utilize this command in Windows terminal:
   ```.venv\Scripts\activate```
   On macOS/Linux terminal:
   ```source .venv/bin/activate```

4. Install Dependencies

   ```pip install -r requirements.txt```
   Ensure requirements.txt includes necessary libraries.

5. Prepare Data
   
   Ensure the data file supermarket_sales.csv is placed in your data/ directory.

# Running the Project

   You will be running the project primarily through a main.py file. However, you are also able to alter projected values by changing the end_date in predict.py (By default, I have the set end date to be in January 2030, but you are able to add additional years and months if needed through changing the datetime value in the script).
