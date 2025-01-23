import pandas as pd

# Load the provided file to examine the data structure
file_path = 'csv/정신질환ment.csv'
data = pd.read_csv(file_path)

# Display the first few rows to understand its structure
data.head()

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Clean and process the data
data_cleaned = data.set_index('AREA').T.reset_index()
data_cleaned.columns = ['Year'] + list(data['AREA'])

# Remove commas and convert to numeric
for col in data_cleaned.columns[1:]:
    data_cleaned[col] = data_cleaned[col].replace(',', '', regex=True).astype(float)

# Calculate yearly averages across all areas
data_cleaned['Average'] = data_cleaned.iloc[:, 1:].mean(axis=1)
data_cleaned['Year'] = data_cleaned['Year'].str.replace(',', '').astype(int)

# Linear regression for prediction
X = data_cleaned['Year'].values.reshape(-1, 1)
y = data_cleaned['Average'].values

model = LinearRegression()
model.fit(X, y)

# Predict for 2024 to 2075
future_years = np.arange(2024, 2076).reshape(-1, 1)
future_predictions = model.predict(future_years)

# Combine years and predictions into a dataframe
predictions = pd.DataFrame({
    'Year': future_years.flatten(),
    'Predicted Average Cost': future_predictions
})

# Visualize the trend
plt.figure(figsize=(10, 6))
plt.scatter(data_cleaned['Year'], data_cleaned['Average'], color='blue', label='Historical Data')
plt.plot(future_years, future_predictions, color='red', linestyle='--', label='Prediction')
plt.title('Private Education Cost Prediction (2024-2075)')
plt.xlabel('Year')
plt.ylabel('Average Cost')
plt.legend()
plt.grid(True)
plt.show()

# Correct column names by removing commas and whitespace
data_cleaned.columns = ['Year'] + [col.replace(',', '') for col in data_cleaned.columns[1:]]

# Remove 'ADD_UP' from the list of regions (areas) and continue with the prediction
regions_to_predict = [col for col in data_cleaned.columns[1:-1] if col != 'ADD_UP']  # Exclude 'ADD_UP'

# Retry the regional prediction process for the remaining regions (excluding 'ADD_UP')
regional_predictions = {}

for region in regions_to_predict:
    regional_data = data_cleaned[['Year', region]].dropna()
    X_region = regional_data['Year'].values.reshape(-1, 1)
    y_region = regional_data[region].values

    # Train a linear regression model for each region
    model_region = LinearRegression()
    model_region.fit(X_region, y_region)

    # Predict for future years
    future_predictions_region = model_region.predict(future_years)
    regional_predictions[region] = future_predictions_region

# Combine years and regional predictions into a DataFrame
predictions_df = pd.DataFrame(future_years, columns=['Year'])

# Add each region's predictions to the DataFrame
for region, predictions in regional_predictions.items():
    predictions_df[region] = predictions

# Transpose the DataFrame so that regions are rows and years are columns
transposed_predictions_df = predictions_df.set_index('Year').T.reset_index()

# Rename 'index' to 'Region' for clarity
transposed_predictions_df.rename(columns={'index': 'Region'}, inplace=True)

# Save the transposed dataframe as a CSV file
output_path_transposed = '정신질환_2024_2075.csv'
transposed_predictions_df.to_csv(output_path_transposed, index=False)

output_path_transposed
