import pandas as pd

# Load the dataset
df = pd.read_csv("../data/train.csv")

print("Original Shape:", df.shape)

# -----------------------------
# 1. Remove duplicate rows
# -----------------------------
df = df.drop_duplicates()

print("After Removing Duplicates:", df.shape)

# -----------------------------
# 2. Handle missing values
# -----------------------------

# Fill numerical columns with the median
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill categorical columns with the mode
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------
# 3. Fix data types
# -----------------------------

# Convert object columns to category
for col in cat_cols:
    df[col] = df[col].astype("category")

print("\nData Types:")
print(df.dtypes)

print("\nRemaining Missing Values:")
print(df.isnull().sum().sum())

# -----------------------------
# 4. Save cleaned dataset
# -----------------------------
df.to_csv("../data/processed/cleaned_train.csv", index=False)

print("\nCleaned dataset saved successfully!")