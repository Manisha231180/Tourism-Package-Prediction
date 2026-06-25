
# Import Hugging Face API utilities
from huggingface_hub import HfApi
import os

# Authenticate using the Hugging Face token stored in environment variables
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload deployment files to the Hugging Face Space
api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id="Manisha231180/Tourism-Package-Prediction",
    repo_type="space",
    path_in_repo=""
)
