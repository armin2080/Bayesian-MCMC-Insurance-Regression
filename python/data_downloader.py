import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("harshsingh2209/medical-insurance-payout")

if not os.path.exists("datasets"):
    os.makedirs("datasets")

# Move contents of the downloaded directory directly to datasets
for item in os.listdir(path):
    item_path = os.path.join(path, item)
    shutil.move(item_path, "datasets/")

os.rmdir(path)

path = "datasets"

print("Path to dataset files:", path)