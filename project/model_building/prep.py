import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from huggingface_hub import HfApi

# Initialize Hugging Face API
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset
DATASET_PATH = "hf://datasets/zezkcy/Tour-Package/tourism.csv"  # local dataset
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unwanted columns
df.drop(columns=["CustomerID", 'Unnamed: 0'], inplace=True, errors="ignore")

# Clean categorical data: lowercase + strip spaces
categorical_cols = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation", "CityTier"]
for col in categorical_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()

# Fix inconsistent gender values
df["Gender"] = df["Gender"].replace({"fe male": "female"})

# Label encoding for categorical columns
label_encoder = LabelEncoder()
for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

# Define target column
target_col = "ProdTaken"

# Features (X) and target (y)
X = df.drop(columns=[target_col])
y = df[target_col]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save splits inside project/data
Xtrain.to_csv("project/data/Xtrain.csv", index=False)
Xtest.to_csv("project/data/Xtest.csv", index=False)
ytrain.to_csv("project/data/ytrain.csv", index=False)
ytest.to_csv("project/data/ytest.csv", index=False)

print("Data split and saved successfully inside project/data folder.")

# Upload files to Hugging Face dataset repo
files = [
    "project/data/Xtrain.csv",
    "project/data/Xtest.csv",
    "project/data/ytrain.csv",
    "project/data/ytest.csv",
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="zezkcy/Tour-Package",  # change to your HF repo
        repo_type="dataset",
    )

print("Files uploaded to Hugging Face successfully.")
