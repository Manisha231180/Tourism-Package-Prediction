
# Import required libraries for Hugging Face dataset management
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

# Retrieve Hugging Face token from environment variables
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define Hugging Face dataset repository details
repo_id = "Manisha231180/Tourism-Package-Prediction"
repo_type = "dataset"

# Check whether the dataset repository already exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset repository '{repo_id}' already exists. Using it.")

except RepositoryNotFoundError:
    print(f"Dataset repository '{repo_id}' not found. Creating it...")
    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=False
    )
    print(f"Dataset repository '{repo_id}' created successfully.")

# Upload all files from the data directory
api.upload_folder(
    folder_path="tourism_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)

print("Dataset files uploaded successfully.")
