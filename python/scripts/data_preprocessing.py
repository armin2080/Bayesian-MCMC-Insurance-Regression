import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_data(filepath='../../data/expenses.csv'):
    return pd.read_csv(filepath)


def check_missing_values(df):
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing)
    return missing


def check_data_types(df):
    print("\nData Types:")
    print(df.dtypes)
    print("\nDataset Info:")
    print(df.info())


def encode_binary_variables(df):
    df_clean = df.copy()
    df_clean['sex'] = (df_clean['sex'] == 'male').astype(int)
    df_clean['smoker'] = (df_clean['smoker'] == 'yes').astype(int)
    return df_clean


def remove_duplicates(df):
    n_duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {n_duplicates}")
    
    df_clean = df.drop_duplicates()
    print(f"Rows after removing duplicates: {len(df_clean)}")
    
    return df_clean


def plot_boxplots(df, output_dir='../../plots'):
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Create figure
    n_cols = len(numeric_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(4*n_cols, 5))
    
    if n_cols == 1:
        axes = [axes]
    
    for idx, col in enumerate(numeric_cols):
        axes[idx].boxplot(df[col].dropna())
        axes[idx].set_title(col)
        axes[idx].set_ylabel('Value')
    
    plt.suptitle('Boxplots of All Numeric Features', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / 'numeric_features_boxplots.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nBoxplot saved to: {output_path}")


def count_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = ((series < lower_bound) | (series > upper_bound)).sum()
    return outliers


def analyze_outliers(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_counts = {}
    
    print("\nOutlier counts per numeric feature:")
    for col in numeric_cols:
        count = count_outliers(df[col])
        outlier_counts[col] = count
        print(f"  {col}: {count}")
    
    return outlier_counts


def normalize_numeric_features(df, features_to_normalize=['age', 'bmi', 'children']):
    df_normalized = df.copy()
    
    for feature in features_to_normalize:
        if feature in df.columns:
            df_normalized[feature] = (df[feature] - df[feature].mean()) / df[feature].std()
            print(f"Normalized {feature}: mean={df_normalized[feature].mean():.6f}, std={df_normalized[feature].std():.6f}")
    
    return df_normalized


def preprocess_data(input_path='../../data/expenses.csv', 
                   output_path='../../data/expenses_cleaned.csv',
                   plot_dir='../outputs'):
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # (1) Load data
    print("\n(1) Loading data...")
    df = load_data(input_path)
    print(f"Dataset shape: {df.shape}")
    
    # (2) Check missing values
    print("\n(2) Checking for missing values...")
    check_missing_values(df)
    
    # (3) Check data types
    print("\n(3) Checking data types...")
    check_data_types(df)
    
    # (4) Encode binary variables
    print("\n(4) Encoding binary variables...")
    df_clean = encode_binary_variables(df)
    print("  - sex: male=1, female=0")
    print("  - smoker: yes=1, no=0")
    
    # (5) Remove duplicates
    print("\n(5) Removing duplicates...")
    df_clean = remove_duplicates(df_clean)
    
    # (6) Analyze outliers
    print("\n(6) Analyzing outliers...")
    plot_boxplots(df_clean, output_dir=plot_dir)
    analyze_outliers(df_clean)
    
    # (7) Normalize numeric features
    print("\n(7) Normalizing numeric features...")
    df_clean = normalize_numeric_features(df_clean)
    
    # (8) Display summary statistics
    print("\n(8) Summary statistics:")
    print(df_clean.describe())
    
    # (9) Save cleaned data
    print(f"\n(9) Saving cleaned data to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    return df_clean


if __name__ == "__main__":
    # Run the preprocessing pipeline
    expenses_clean = preprocess_data()
    
    print("\n\nCleaned dataset preview:")
    print(expenses_clean.head())
    print(f"\nFinal shape: {expenses_clean.shape}")
