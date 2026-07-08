"""
Advanced Feature 2

Predict a house price from custom user input.
"""

import joblib
import pandas as pd

MODEL_PATH = "../models/house_price_model.pkl"

try:
    model = joblib.load(MODEL_PATH)

    # Load training columns
    X_train = pd.read_csv("../data/processed/X_train.csv")

    # Use first row as a template
    sample = X_train.iloc[[0]].copy()

    print("\nEnter House Details\n")

    area = float(input("Living Area (sq ft): "))
    quality = int(input("Overall Quality (1-10): "))
    garage = int(input("Garage Cars: "))

    # Update sample values
    if "GrLivArea" in sample.columns:
        sample["GrLivArea"] = area

    if "OverallQual" in sample.columns:
        sample["OverallQual"] = quality

    if "GarageCars" in sample.columns:
        sample["GarageCars"] = garage

    prediction = model.predict(sample)

    print("\n=========================")
    print(f"Estimated Price: ₹{prediction[0]:,.2f}")
    print("=========================")

except ValueError:
    print("Please enter valid numeric values.")

except Exception as e:
    print("Error:", e)