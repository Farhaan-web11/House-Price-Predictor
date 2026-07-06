import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("../data/processed/cleaned_train.csv")

# -------------------------------
# Feature Engineering
# -------------------------------

# Feature 1: Total Bathrooms
df["TotalBathrooms"] = (
    df["FullBath"]
    + 0.5 * df["HalfBath"]
    + df["BsmtFullBath"]
    + 0.5 * df["BsmtHalfBath"]
)

# Feature 2: Total House Area
df["TotalArea"] = df["GrLivArea"] + df["TotalBsmtSF"]

# -------------------------------
# Separate Target
# -------------------------------
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# -------------------------------
# One-Hot Encode Categorical Columns
# -------------------------------
X = pd.get_dummies(X, drop_first=True)

# -------------------------------
# Scale Numerical Features
# -------------------------------
scaler = StandardScaler()

numeric_columns = X.select_dtypes(include=["int64", "float64"]).columns

X[numeric_columns] = scaler.fit_transform(X[numeric_columns])

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Features:", X_train.shape)
print("Testing Features :", X_test.shape)
print("Training Labels  :", y_train.shape)
print("Testing Labels   :", y_test.shape)

# Save processed datasets
X_train.to_csv("../data/processed/X_train.csv", index=False)
X_test.to_csv("../data/processed/X_test.csv", index=False)
y_train.to_csv("../data/processed/y_train.csv", index=False)
y_test.to_csv("../data/processed/y_test.csv", index=False)

print("\nFeature engineering completed successfully!")