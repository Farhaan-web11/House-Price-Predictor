import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../data/train.csv")

# Shape
print("\n===== Dataset Shape =====")
print(df.shape)

# Columns
print("\n===== Columns =====")
print(df.columns)

# Data Types
print("\n===== Data Types =====")
print(df.dtypes)

# Missing Values
print("\n===== Missing Values =====")
print(df.isnull().sum())
# Plot 1 - Sale Price Distribution
plt.figure(figsize=(8,5))
df["SalePrice"].hist(bins=30)
plt.title("Sale Price Distribution")
plt.xlabel("Sale Price")
plt.ylabel("Count")
plt.show()

# Plot 2 - Overall Quality
plt.figure(figsize=(8,5))
df["OverallQual"].hist()
plt.title("Overall Quality")
plt.xlabel("Quality")
plt.ylabel("Count")
plt.show()

# Plot 3 - Living Area
plt.figure(figsize=(8,5))
df["GrLivArea"].hist()
plt.title("Ground Living Area")
plt.xlabel("Area")
plt.ylabel("Count")
plt.show()