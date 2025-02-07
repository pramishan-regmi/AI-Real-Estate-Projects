# model_training.py
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from feature_engineering import X_train_scaled, X_test_scaled, y_train, y_test  # Import preprocessed data
from sklearn.preprocessing import StandardScaler
import os

# Create 'models' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Initialize and fit scaler
scaler = StandardScaler()
scaler.fit(X_train_scaled)

# Train Linear Regression Model
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_preds = lr_model.predict(X_test_scaled)

# Train XGBoost Model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)

# Save the models
joblib.dump(xgb_model, "models/property_price_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
print("Model and scaler saved!")
