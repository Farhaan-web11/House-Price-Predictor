import pandas as pd
import joblib

# Load model
model = joblib.load("../models/house_price_model.pkl")

# Load test data
X_test = pd.read_csv("../data/processed/X_test.csv")

print("First 5 Predictions:\n")

for i in range(5):
    prediction = model.predict([X_test.iloc[i]])[0]
    print(f"House {i+1}: ₹{prediction:,.2f}")