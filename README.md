# Bayesian-MCMC-Insurance-Regression

A statistical analysis project implementing Bayesian MCMC regression models to predict medical insurance costs. This project includes implementations in both Python and R.

## Project Structure

```
├── data/                      # Shared datasets
│   └── expenses.csv          # Medical insurance dataset
├── plots/                     # Visualizations
├── python/                    # Python implementation
│   ├── data_downloader.py    # Script to download dataset from Kaggle
│   ├── notebook.ipynb        # Jupyter notebook with analysis
│   ├── requirements.txt      # Python dependencies
│   └── env/                  # Python virtual environment
└── r/                         # R implementation
    ├── scripts/
    │   ├── Data_Preprocessing.R
    │   └── Model_Setup.R
    └── outputs/              # R analysis outputs
        ├── cor_matrix.txt
        └── ols_output.txt
```

## Getting Started

### Python Setup
```bash
cd python
source env/bin/activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

### R Setup
```bash
cd r/scripts
# Install required R packages, then:
Rscript Data_Preprocessing.R
Rscript Model_Setup.R
```

## Data Overview

This project uses the Medical Insurance Cost dataset from Kaggle, containing information about insurance charges based on various factors.

### Dataset Summary
- **Source**: Kaggle (harshsingh2209/medical-insurance-payout)
- **Rows**: 1,338 (after removing duplicates)
- **Columns**: 7 original features + encoded variables

### Features
- **age**: Age of the insured (18-64)
- **sex**: Gender (male=1, female=0)
- **bmi**: Body Mass Index (15.96-53.13)
- **children**: Number of children (0-5)
- **smoker**: Smoking status (yes=1, no=0)
- **region**: Geographic region (encoded as dummies: northwest, southeast, southwest)
- **charges**: Medical insurance costs (target variable)

### Summary Statistics

| Feature  | Mean     | Std Dev | Min     | Max     |
|----------|----------|---------|---------|---------|
| age      | 39.21    | 14.05   | 18.00   | 64.00   |
| bmi      | 30.66    | 6.10    | 15.96   | 53.13   |
| children | 1.09     | 1.21    | 0.00    | 5.00    |
| charges  | 13270.42 | 12110.01| 1121.87 | 63770.43|

### Categorical Distributions

| sex      | Count | Percentage |
|----------|-------|------------|
| Female   | 662   | 49.5%      |
| Male     | 676   | 50.5%      |

| smoker   | Count | Percentage |
|----------|-------|------------|
| No       | 1064  | 79.5%      |
| Yes      | 274   | 20.5%      |

| region   | Count | Percentage |
|----------|-------|------------|
| Southwest| 325   | 24.3%      |
| Southeast| 364   | 27.2%      |
| Northwest| 325   | 24.3%      |
| Northeast| 324   | 24.2%      |

### Key Insights
- Average insurance charge: $13,270
- Smokers have significantly higher charges (mean ~$32,050 vs $8,434 for non-smokers)
- BMI shows a positive correlation with charges
- Age also positively correlates with charges
- Males have slightly higher average charges than females

### Data Cleaning Performed
- Checked for missing values (none found)
- Removed 1 duplicate entry
- Encoded categorical variables (sex/smoker to binary, region to dummies)
- Standardized numerical features for modeling
- Outliers detected but retained for Bayesian analysis