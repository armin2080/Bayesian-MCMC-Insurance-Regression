#!/usr/bin/env python
"""
Regenerate all plots for the report with 4 chains.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys

from data_preprocessing import preprocess_data
from model_setup import create_design_matrix
from gibbs_sampling import gibbs_lm, beta_trace_plot, sigma2_trace_plot
from metropolis_hastings import metropolis_hastings_lm
from algorithm_comparison import compare_algorithms
from residual_diagnostics import create_residual_diagnostic_plot, create_residual_comparison_plot
from mcmc_diagnostics import rhat, ess_geyer

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("Regenerating all plots with 4 chains")
print("="*80)

df = pd.read_csv('../../data/expenses_cleaned.csv')
print(f"\nData loaded: {df.shape}")

print("\n[1/9] Descriptive statistics...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Insurance Dataset - Descriptive Statistics', fontsize=16, y=1.00)

# Age distribution
axes[0, 0].hist(df['age'], bins=20, edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Age')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title(f'Age Distribution\n(Mean: {df["age"].mean():.1f}, Range: {df["age"].min()}-{df["age"].max()})')

# Sex distribution
sex_counts = df['sex'].value_counts()
axes[0, 1].bar(sex_counts.index, sex_counts.values, edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Sex')
axes[0, 1].set_ylabel('Count')
axes[0, 1].set_title('Sex Distribution')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_xticklabels(['Female', 'Male'])

# BMI distribution
axes[0, 2].hist(df['bmi'], bins=20, edgecolor='black', alpha=0.7, color='green')
axes[0, 2].set_xlabel('BMI')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].set_title(f'BMI Distribution\n(Mean: {df["bmi"].mean():.1f}, Range: {df["bmi"].min():.1f}-{df["bmi"].max():.1f})')

# Children distribution
children_counts = df['children'].value_counts().sort_index()
axes[1, 0].bar(children_counts.index, children_counts.values, edgecolor='black', alpha=0.7, color='orange')
axes[1, 0].set_xlabel('Number of Children')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Children Distribution')

# Smoker distribution
smoker_counts = df['smoker'].value_counts()
axes[1, 1].bar(smoker_counts.index, smoker_counts.values, edgecolor='black', alpha=0.7, color='red')
axes[1, 1].set_xlabel('Smoker Status')
axes[1, 1].set_ylabel('Count')
axes[1, 1].set_title(f'Smoker Distribution\n({(df["smoker"]==1).sum()/len(df)*100:.1f}% smokers)')
axes[1, 1].set_xticks([0, 1])
axes[1, 1].set_xticklabels(['Non-smoker', 'Smoker'])

# Charges distribution
axes[1, 2].hist(df['charges'], bins=30, edgecolor='black', alpha=0.7, color='purple')
axes[1, 2].set_xlabel('Charges ($)')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title(f'Charges Distribution\n(Mean: ${df["charges"].mean():.0f}, Median: ${df["charges"].median():.0f})')

plt.tight_layout()
output_path = Path('../outputs/descriptive_statistics.png')
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved to {output_path}")

# ============================================================================
# 2. BASELINE MODEL - GIBBS SAMPLING
# ============================================================================
print("\n[2/9] Running baseline model - Gibbs Sampling (4 chains)...")

X, y, feature_names = create_design_matrix(df, formula='baseline')
param_names = ['(Intercept)', 'age', 'sex', 'bmi', 'children', 'smoker']

gibbs_results = gibbs_lm(
    y=y, X=X,
    n_iter=10000, warmup=2000, n_chains=4,
    seed=123
)

# Generate trace plots for Gibbs
print("   Generating Gibbs trace plots...")
beta_list = [chain['beta'] for chain in gibbs_results]
sigma2_list = [chain['sigma2'] for chain in gibbs_results]

# Beta trace plots (save to gibbs_result directory)
beta_trace_plot(beta_list, model_name='gibbs_result', plot_dir='../outputs')
print("   ✓ Gibbs trace plots saved to ../outputs/gibbs_result/")

# Residual diagnostics for baseline model
print("   Generating baseline residual diagnostics...")
beta_combined = np.vstack(beta_list)
sigma2_combined = np.concatenate(sigma2_list)

baseline_diag = create_residual_diagnostic_plot(
    y_obs=y,
    X=X,
    beta_samples=beta_combined,
    sigma2_samples=sigma2_combined,
    model_name='baseline_model',
    output_dir='../outputs'
)
print("   ✓ Baseline residual diagnostics saved")

# ============================================================================
# 3. BASELINE MODEL - METROPOLIS-HASTINGS
# ============================================================================
print("\n[3/9] Running baseline model - Metropolis-Hastings (4 chains)...")

mh_results = metropolis_hastings_lm(
    y=y, X=X,
    n_iter=10000, warmup=2000, n_chains=4,
    proposal_sd_beta=0.0001, proposal_sd_sigma2=0.4,
    seed=456
)

print("   MH Acceptance Rates:")
for i, result in enumerate(mh_results, 1):
    print(f"     Chain {i}: Beta={result.get('beta_acceptance', 0):.3f}, "
          f"Sigma2={result.get('sigma2_acceptance', 0):.3f}")

# Generate trace plots for MH
print("   Generating MH trace plots...")
mh_beta_list = [chain['beta'] for chain in mh_results]
mh_sigma2_list = [chain['sigma2'] for chain in mh_results]

# Beta trace plots (save to mh_baseline directory)
mh_output_dir = Path('../outputs/mh_baseline')
mh_output_dir.mkdir(parents=True, exist_ok=True)

# Generate individual trace plots for each beta parameter
for j in range(mh_beta_list[0].shape[1]):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10))
    fig.suptitle(f'MH Trace Plot - {param_names[j]}', fontsize=14)
    
    for chain_idx, chain_beta in enumerate(mh_beta_list):
        axes[chain_idx].plot(chain_beta[:, j], alpha=0.7, linewidth=0.5)
        axes[chain_idx].set_ylabel(f'Chain {chain_idx+1}')
        axes[chain_idx].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Iteration')
    plt.tight_layout()
    plt.savefig(mh_output_dir / f'beta_trace_{j+1}.png', dpi=150, bbox_inches='tight')
    plt.close()

print("   ✓ MH trace plots saved to ../outputs/mh_baseline/")

# ============================================================================
# 4. ALGORITHM COMPARISON
# ============================================================================
print("\n[4/9] Generating algorithm comparison...")

compare_algorithms(
    gibbs_results=gibbs_results,
    mh_results=mh_results,
    param_names=param_names,
    output_dir='../outputs/algorithm_comparison'
)

# Also generate summary plot explicitly
from algorithm_comparison import create_comparison_summary_plot
create_comparison_summary_plot(
    gibbs_results=gibbs_results,
    mh_results=mh_results,
    param_names=param_names,
    output_dir='../outputs/algorithm_comparison'
)

print("   ✓ Algorithm comparison plots saved")

# ============================================================================
# 5. LOG-TRANSFORMED MODEL
# ============================================================================
print("\n[5/9] Running log-transformed model (4 chains)...")

X_log, y_log, feature_names_log = create_design_matrix(df, formula='log')

log_results = gibbs_lm(
    y=y_log, X=X_log,
    n_iter=10000, warmup=2000, n_chains=4,
    seed=789
)

# Residual diagnostics for log model
print("   Generating log-transformed residual diagnostics...")
beta_log_combined = np.vstack([chain['beta'] for chain in log_results])
sigma2_log_combined = np.concatenate([chain['sigma2'] for chain in log_results])

log_diag = create_residual_diagnostic_plot(
    y_obs=y_log,
    X=X_log,
    beta_samples=beta_log_combined,
    sigma2_samples=sigma2_log_combined,
    model_name='log_transformed',
    output_dir='../outputs'
)
print("   ✓ Log-transformed residual diagnostics saved")

# ============================================================================
# 6. INTERACTION MODEL
# ============================================================================
print("\n[6/9] Running interaction model (4 chains)...")

X_int, y_int, feature_names_int = create_design_matrix(df, formula='interaction')

int_results = gibbs_lm(
    y=y_int, X=X_int,
    n_iter=10000, warmup=2000, n_chains=4,
    seed=101112
)

# Residual diagnostics for interaction model
print("   Generating interaction model residual diagnostics...")
beta_int_combined = np.vstack([chain['beta'] for chain in int_results])
sigma2_int_combined = np.concatenate([chain['sigma2'] for chain in int_results])

int_diag = create_residual_diagnostic_plot(
    y_obs=y_int,
    X=X_int,
    beta_samples=beta_int_combined,
    sigma2_samples=sigma2_int_combined,
    model_name='interaction',
    output_dir='../outputs'
)
print("   ✓ Interaction residual diagnostics saved")

# ============================================================================
# 7. RESIDUAL COMPARISON PLOT
# ============================================================================
print("\n[7/9] Generating residual comparison plot...")

diagnostics_dict = {
    'Baseline': baseline_diag,
    'Log-Transformed': log_diag,
    'Interaction': int_diag
}

create_residual_comparison_plot(
    diagnostics_dict=diagnostics_dict,
    output_dir='../outputs'
)
print("   ✓ Residual comparison plot saved")

# ============================================================================
# 8. VERIFY ALL PLOTS EXIST
# ============================================================================
print("\n[8/9] Verifying all plots exist...")

required_plots = [
    '../outputs/descriptive_statistics.png',
    '../outputs/baseline_model/residual_diagnostics.png',
    '../outputs/log_transformed/residual_diagnostics.png',
    '../outputs/interaction/residual_diagnostics.png',
    '../outputs/residual_comparison.png',
    '../outputs/gibbs_result/beta_trace_5.png',  # Smoker coefficient (index 5)
    '../outputs/mh_baseline/beta_trace_6.png',   # Smoker coefficient (index 5, but file numbered 6)
    '../outputs/algorithm_comparison/autocorrelation_comparison.png',
    '../outputs/algorithm_comparison/algorithm_comparison_summary.png',
]

missing_plots = []
for plot_path in required_plots:
    full_path = Path(plot_path)
    if not full_path.exists():
        missing_plots.append(plot_path)
        print(f"   ✗ MISSING: {plot_path}")
    else:
        print(f"   ✓ EXISTS: {plot_path}")

# ============================================================================
# 9. SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PLOT REGENERATION COMPLETE")
print("="*80)

if missing_plots:
    print(f"\n⚠ Warning: {len(missing_plots)} plot(s) are missing:")
    for plot in missing_plots:
        print(f"  - {plot}")
else:
    print("\n✓ All required plots have been generated successfully!")
    print("\nPlots generated with:")
    print("  - 4 chains")
    print("  - 10,000 iterations")
    print("  - 2,000 burn-in")
    print("  - 32,000 total samples per algorithm")

print("\n" + "="*80)
