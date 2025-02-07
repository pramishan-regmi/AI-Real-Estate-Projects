# data_collection.py
from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["Price"] = data.target * 100000  # Convert to real dollar values

# Create 'data' directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save dataset
df.to_csv("data/california_housing.csv", index=False)
print("Dataset saved!")
