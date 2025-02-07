# feature_engineering.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/california_housing.csv")

# Create new features
df["Rooms_per_Household"] = df["AveRooms"] / df["HouseAge"]
df["Bedrooms_per_Room"] = df["AveBedrms"] / df["AveRooms"]
df["Population_per_Household"] = df["Population"] / df["HouseAge"]

# Drop redundant columns
df.drop(columns=["AveRooms", "AveBedrms"], inplace=True)

# Split into training & test sets
X = df.drop(columns=["Price"])
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Feature Engineering Done!")
