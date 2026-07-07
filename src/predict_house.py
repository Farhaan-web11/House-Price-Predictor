"""
Feature 3: House Price Prediction

This script:
1. Loads the trained model.
2. Uses sample house data.
3. Predicts the house price.
"""

import joblib
import pandas as pd

# Load trained model
model = joblib.load("../models/house_price_model.pkl")

# Load training columns to match the model
X_train = pd.read_csv("../data/processed/X_train.csv")

# Create a sample input using the first row
sample_house = X_train.iloc[[0]].copy()

# Predict the price
predicted_price = model.predict(sample_house)

print("=" * 40)
print("HOUSE PRICE PREDICTION")
print("=" * 40)
print(f"Predicted House Price: ₹{predicted_price[0]:,.2f}")
print("=" * 40)