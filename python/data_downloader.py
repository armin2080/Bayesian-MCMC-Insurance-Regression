import kagglehub
import shutil
import os

path = kagglehub.dataset_download("harshsingh2209/medical-insurance-payout")

if not os.path.exists("datasets"):
    os.makedirs("datasets")

for item in os.listdir(path):
    item_path = os.path.join(path, item)
    shutil.move(item_path, "datasets/")

os.rmdir(path)
path = "datasets"

print("Path to dataset files:", path)