from huggingface_hub import HfApi
import os

# Initialize API with authentication token
api = HfApi(token=os.getenv("HF_TOKEN"))

# Upload deployment folder to Hugging Face Space
# Replace with your Hugging Face username
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="ssowmiya/Tourism-Package-Prediction",  # the target Hugging Face Space
    repo_type="space",                            # repository type: space
    path_in_repo="",                              # optional: subfolder path inside the repo
)

print("Deployment files uploaded to Hugging Face Space successfully!")
