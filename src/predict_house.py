"""
House Price Prediction with Input Validation
"""

import os
import joblib
import pandas as pd

MODEL_PATH = "../models/house_price_model.pkl"
DATA_PATH = "../data/processed/X_train.csv"

try:
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Run train_model.py first.")

    # Check if dataset exists
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Processed training data not found.")

    print("Loading model...")
    model = joblib.load(MODEL_PATH)

    print("Loading sample data...")
    X_train = pd.read_csv(DATA_PATH)

    # Validate dataset
    if X_train.empty:
        raise ValueError("Dataset is empty.")

    # Take first sample
    sample = X_train.iloc[[0]]

    print("Making prediction...")
    prediction = model.predict(sample)

    print("\n========== RESULT ==========")
    print(f"Predicted House Price: ₹{prediction[0]:,.2f}")
    print("============================")

except FileNotFoundError as e:
    print(f"File Error: {e}")

except ValueError as e:
    print(f"Validation Error: {e}")

except Exception as e:
    print(f"Unexpected Error: {e}")