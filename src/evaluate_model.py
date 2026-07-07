"""
Feature 2: Model Evaluation

This script:
1. Loads the trained model.
2. Makes predictions.
3. Evaluates model performance.
4. Saves prediction results.
"""

import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

try:
    # Load trained model
    model = joblib.load("../models/house_price_model.pkl")

    # Load test data
    X_test = pd.read_csv("../data/processed/X_test.csv")
    y_test = pd.read_csv("../data/processed/y_test.csv").squeeze()

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("Model Evaluation")
    print("----------------")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")

    # Save prediction results
    results = pd.DataFrame({
        "Actual Price": y_test,
        "Predicted Price": predictions
    })

    os.makedirs("../results", exist_ok=True)
    results.to_csv("../results/predictions.csv", index=False)

    # Plot comparison
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")
    plt.grid(True)

    plt.savefig("../results/actual_vs_predicted.png")
    plt.show()

    print("Results saved in the results folder.")

except FileNotFoundError:
    print("Error: Required file not found.")

except Exception as e:
    print("Unexpected Error:", e)