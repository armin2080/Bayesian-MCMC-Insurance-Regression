# ==============================================================================
# Data Downloader for Medical Insurance Dataset
# ==============================================================================
# This script downloads the medical insurance payout dataset from Kaggle
# using the kaggle API.
#
# Prerequisites:
# 1. Install kaggle CLI: pip install kaggle
# 2. Setup Kaggle API credentials:
#    - Download kaggle.json from https://www.kaggle.com/settings
#    - Place it in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%/.kaggle/ (Windows)
#    - Set permissions: chmod 600 ~/.kaggle/kaggle.json
# ==============================================================================

cat("=", 80, "\n", sep="")
cat("DOWNLOADING MEDICAL INSURANCE DATASET FROM KAGGLE\n")
cat("=", 80, "\n\n", sep="")

# Dataset information
dataset_slug <- "harshsingh2209/medical-insurance-payout"
target_dir <- "../../data"

cat(sprintf("Dataset: %s\n", dataset_slug))
cat(sprintf("Target directory: %s\n\n", target_dir))

# Create target directory if it doesn't exist
if (!dir.exists(target_dir)) {
  dir.create(target_dir, recursive = TRUE)
  cat(sprintf("✓ Created directory: %s\n", target_dir))
}

# Check if kaggle CLI is available
kaggle_available <- system("which kaggle", ignore.stdout = TRUE, ignore.stderr = TRUE) == 0

if (!kaggle_available) {
  cat(" Kaggle CLI not found!\n\n")
  cat("To install:\n")
  cat("  1. Install Python and pip\n")
  cat("  2. Run: pip install kaggle\n")
  cat("  3. Download kaggle.json from https://www.kaggle.com/settings\n")
  cat("  4. Place it in ~/.kaggle/ and set permissions: chmod 600 ~/.kaggle/kaggle.json\n\n")
  cat("Alternatively, manual download:\n")
  cat(sprintf("  Visit: https://www.kaggle.com/datasets/%s\n", dataset_slug))
  cat(sprintf("  Download and extract to: %s\n", target_dir))
  stop("Kaggle CLI not available")
}

cat(" Kaggle CLI found\n\n")

# Download dataset
cat("Downloading dataset...\n")

# Use system2 to run kaggle command
result <- system2(
  command = "kaggle",
  args = c("datasets", "download", "-d", dataset_slug, "-p", target_dir, "--unzip"),
  stdout = TRUE,
  stderr = TRUE
)

if (attr(result, "status") == 0 || is.null(attr(result, "status"))) {
  cat(" Download complete!\n\n")
  
  # List downloaded files
  cat("Downloaded files:\n")
  files <- list.files(target_dir, full.names = FALSE)
  for (file in files) {
    file_path <- file.path(target_dir, file)
    if (file.exists(file_path)) {
      size_mb <- file.info(file_path)$size / (1024^2)
      cat(sprintf("  - %s (%.2f MB)\n", file, size_mb))
    }
  }
  
  cat("\n")
  cat("=", 80, "\n", sep="")
  cat("DATASET READY FOR ANALYSIS\n")
  cat("=", 80, "\n", sep="")
  
} else {
  cat(" Download failed!\n")
  cat("\nError output:\n")
  cat(paste(result, collapse = "\n"))
  cat("\n\nTroubleshooting:\n")
  cat("  1. Check Kaggle API credentials: ~/.kaggle/kaggle.json\n")
  cat("  2. Verify dataset URL: https://www.kaggle.com/datasets/%s\n", dataset_slug)
  cat("  3. Ensure you've accepted the dataset's terms on Kaggle website\n")
  cat("\nManual download option:\n")
  cat(sprintf("  Visit: https://www.kaggle.com/datasets/%s\n", dataset_slug))
  cat(sprintf("  Download and extract to: %s\n", target_dir))
}
