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


def drop_region(df):
    """Drop region column since we're not doing hierarchical modeling."""
    df_clean = df.copy()
    if 'region' in df_clean.columns:
        df_clean = df_clean.drop('region', axis=1)
        print("  - Dropped 'region' column")
    return df_clean


def remove_duplicates(df):
    n_duplicates = df.duplicated().sum()
    print(f"\nNumber of duplicate rows: {n_duplicates}")
    
    df_clean = df.drop_duplicates()
    print(f"Rows after removing duplicates: {len(df_clean)}")
    
    return df_clean


def plot_boxplots(df, output_dir='../../plots'):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
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
    print("Preprocessing data")
    print("=" * 60)
    
    print("\nLoading data...")
    df = load_data(input_path)
    print(f"Dataset shape: {df.shape}")
    
    print("\nChecking for missing values...")
    check_missing_values(df)
    
    print("\nChecking data types...")
    check_data_types(df)
    
    print("\nEncoding binary variables...")
    df_clean = encode_binary_variables(df)
    print("  - sex: male=1, female=0")
    print("  - smoker: yes=1, no=0")
    
    print("\nDropping region column...")
    df_clean = drop_region(df_clean)
    
    print("\nRemoving duplicates...")
    df_clean = remove_duplicates(df_clean)
    
    print("\nAnalyzing outliers...")
    plot_boxplots(df_clean, output_dir=plot_dir)
    analyze_outliers(df_clean)
    
    print("\nNormalizing numeric features...")
    df_clean = normalize_numeric_features(df_clean)
    
    print("\nSummary statistics:")
    print(df_clean.describe())
    
    print(f"\nSaving cleaned data to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    print("\nDone.")
    
    return df_clean


if __name__ == "__main__":
    expenses_clean = preprocess_data()
    
    print("\n\nCleaned dataset preview:")
    print(expenses_clean.head())
    print(f"\nFinal shape: {expenses_clean.shape}")
