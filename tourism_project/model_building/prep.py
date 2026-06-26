
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
import os

# Authenticate with Hugging Face
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset
df = pd.read_csv("tourism_project/data/Tourism-Package-Prediction.csv")

# Remove missing values
df = df.dropna()

# Remove non-predictive columns
df = df.drop(columns=["Unnamed: 0", "CustomerID"])

# Target variable
target_column = "ProdTaken"

# Features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save processed datasets
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload processed datasets to Hugging Face
repo_id = "Manisha231180/Tourism-Package-Prediction"

for file_name in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name,
        repo_id=repo_id,
        repo_type="dataset"
    )

print("Train-test datasets uploaded successfully.")
