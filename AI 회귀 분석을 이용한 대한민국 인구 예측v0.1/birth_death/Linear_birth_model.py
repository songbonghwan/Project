import pandas as pd

# Load the data to check its structure
file_path = '/mnt/data/시군구_출생아수_full.csv'
birth_data = pd.read_csv(file_path)

# Display the first few rows to understand the structure of the data
birth_data.head()

from sklearn.linear_model import LinearRegression
import numpy as np

# Extracting the relevant data for 'ADD_UP' (total births) and preparing it for modeling
add_up_data = birth_data[birth_data['AREA'] == 'ADD_UP']
years = np.array(range(2007, 2024)).reshape(-1, 1)  # Years as features (2007-2023)
births = add_up_data.iloc[0, 1:].values.reshape(-1, 1).astype(float)  # Birth counts as target

# Fit the Linear Regression model
model = LinearRegression()
model.fit(years, births)

# Generate future years for prediction (2024-2075)
future_years = np.array(range(2024, 2076)).reshape(-1, 1)
predicted_births = model.predict(future_years)

# Combine the years and predictions into a DataFrame
prediction_results = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted_Births": predicted_births.flatten()
})

# Get all unique areas excluding the 'ADD_UP' row
areas = birth_data['AREA'].unique()
areas = [area for area in areas if area != 'ADD_UP']

# Create an empty DataFrame to store predictions for each area
area_predictions = pd.DataFrame()

# Predict future births for each area
for area in areas:
    # Filter data for the specific area
    area_data = birth_data[birth_data['AREA'] == area]
    area_years = np.array(range(2007, 2024)).reshape(-1, 1)
    area_births = area_data.iloc[0, 1:].values.reshape(-1, 1).astype(float)
    
    # Train a Linear Regression model
    area_model = LinearRegression()
    area_model.fit(area_years, area_births)
    
    # Predict for future years
    area_predicted_births = area_model.predict(future_years)
    
    # Append predictions to the main DataFrame
    area_predictions = pd.concat([
        area_predictions,
        pd.DataFrame({
            "Year": future_years.flatten(),
            "Area": area,
            "Predicted_Births": area_predicted_births.flatten()
        })
    ])

# Adjust predictions to ensure no negative values by modifying the slope (manually setting a minimum of 0)
adjusted_predictions = pd.DataFrame()

for area in areas:
    # Filter data for the specific area
    area_data = birth_data[birth_data['AREA'] == area]
    area_years = np.array(range(2007, 2024)).reshape(-1, 1)
    area_births = area_data.iloc[0, 1:].values.reshape(-1, 1).astype(float)
    
    # Train a Linear Regression model
    area_model = LinearRegression()
    area_model.fit(area_years, area_births)
    
    # Predict for future years with a slope adjustment if necessary
    area_predicted_births = area_model.predict(future_years)
    area_predicted_births = np.maximum(area_predicted_births, 0)  # Set a minimum of 0 to avoid negative values
    
    # Append adjusted predictions to the main DataFrame
    adjusted_predictions = pd.concat([
        adjusted_predictions,
        pd.DataFrame({
            "Year": future_years.flatten(),
            "Area": area,
            "Predicted_Births": area_predicted_births.flatten()
        })
    ])
# Pivot the data to have regions as columns and years as rows
pivoted_predictions = adjusted_predictions.pivot(index="Year", columns="Area", values="Predicted_Births")

# Save to CSV file
output_path = "/mnt/data/Adjusted_Regional_Predicted_Births.csv"
pivoted_predictions.to_csv(output_path)

output_path

# Reverse the pivot: Make years as columns and areas as rows
reversed_predictions = pivoted_predictions.transpose()

# Save to a new CSV file
reversed_output_path = "/mnt/data/Reversed_Regional_Predicted_Births.csv"
reversed_predictions.to_csv(reversed_output_path)

reversed_output_path