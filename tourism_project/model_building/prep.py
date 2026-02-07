# Import necessary libraries
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Replace with your Hugging Face username
DATASET_PATH = "hf://datasets/ssowmiya/Tourism-Package-Prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Original dataset shape: {df.shape}")

# Data Cleaning
# Drop the unnamed index column and CustomerID (unique identifier, not useful for prediction)
columns_to_drop = ['CustomerID']
if 'Unnamed: 0' in df.columns:
    columns_to_drop.append('Unnamed: 0')
df.drop(columns=columns_to_drop, inplace=True)

# Handle missing values
# Fill numerical columns with median
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Fill categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].mode()[0], inplace=True)

# Fix Gender inconsistencies (e.g., 'Fe Male' -> 'Female')
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].replace({'Fe Male': 'Female'})

print(f"Cleaned dataset shape: {df.shape}")
print(f"Missing values after cleaning: {df.isnull().sum().sum()}")

# Define target column
target_col = 'ProdTaken'

# Split into X (features) and y (target)
X = df.drop(columns=[target_col])
y = df[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {Xtrain.shape[0]}")
print(f"Test set size: {Xtest.shape[0]}")

# Save the split datasets locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload the split datasets to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="ssowmiya/Tourism-Package-Prediction",
        repo_type="dataset",
    )
    print(f"Uploaded {file_path} to Hugging Face Hub")

print("Data preparation completed successfully!")
