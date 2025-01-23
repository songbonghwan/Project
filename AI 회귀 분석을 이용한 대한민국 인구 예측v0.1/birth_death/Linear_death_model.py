import pandas as pd

# Load the uploaded file to inspect its contents
file_path = '/mnt/data/사망자수_ADD_full.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()

from sklearn.linear_model import LinearRegression
import numpy as np

# Data preprocessing for ADD_UP
data['ADD_UP'] = data['AREA'] == 'ADD_UP'
add_up_data = data[data['ADD_UP']].iloc[:, 1:].replace(',', '', regex=True).astype(float)

# Preparing the dataset for linear regression
years = np.array(range(2007, 2024))  # Existing years in the dataset
deaths = add_up_data.iloc[0, :len(years)].values  # Corresponding deaths

# Reshape years for model input
X = years.reshape(-1, 1)
y = deaths

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict deaths for 2024 to 2075
future_years = np.array(range(2024, 2076)).reshape(-1, 1)
predicted_deaths = model.predict(future_years)

# Combine years with predictions into a DataFrame for clarity
predictions_df = pd.DataFrame({
    'Year': range(2024, 2076),
    'Predicted_Deaths': predicted_deaths
})

# Extract data for all regions
regions = data['AREA'].unique()
regional_predictions = {}

# Iterate through each region and predict future deaths
for region in regions:
    region_data = data[data['AREA'] == region].iloc[:, 1:].replace(',', '', regex=True).astype(float)
    deaths = region_data.iloc[0, :len(years)].values  # Existing data
    
    # Train a new model for each region
    model = LinearRegression()
    model.fit(X, deaths)
    
    # Predict for 2024 to 2075
    predicted_deaths = model.predict(future_years)
    regional_predictions[region] = predicted_deaths

# Combine predictions into a DataFrame for easy comparison
regional_predictions_df = pd.DataFrame(future_years.flatten(), columns=['Year'])
for region, predictions in regional_predictions.items():
    regional_predictions_df[region] = predictions

# Transpose the DataFrame to have regions as columns and years as rows
transposed_df = regional_predictions_df.set_index('Year').T

# Save the transposed DataFrame as a CSV file
transposed_file_path = '/mnt/data/Regional_Predicted_Deaths_Transposed.csv'
transposed_df.to_csv(transposed_file_path)

transposed_file_path