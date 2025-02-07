# model_evaluation.py
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from model_training import lr_preds, xgb_preds  # Assuming predictions are in model_training.py
from feature_engineering import y_test  # Import y_test from feature_engineering

import numpy as np

def evaluate_model(name, y_true, y_pred):
    print(f"📊 {name} Performance:")
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # Manually calculating RMSE
    r2 = r2_score(y_true, y_pred)
    
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")  # Now using the manually computed RMSE
    print(f"R² Score: {r2:.2f}")
    print("-" * 30)

# Evaluate models
evaluate_model("Linear Regression", y_test, lr_preds)
evaluate_model("XGBoost", y_test, xgb_preds)
