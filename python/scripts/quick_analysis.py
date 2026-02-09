#!/usr/bin/env python
"""Quick analysis runner for generating fresh results without region"""

import numpy as np
import pandas as pd
from pathlib import Path
from gibbs_sampling import gibbs_lm

print("="*80)
print("RUNNING GIBBS SAMPLING ON CLEANED DATA (NO REGION)")
print("="*80)

# Load cleaned data (without region)
df = pd.read_csv('../../data/expenses_cleaned.csv')
print(f"\nData shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Prepare data
y = df['charges'].values
X = df.drop('charges', axis=1).values
X_with_intercept = np.column_stack([np.ones(len(y)), X])

feature_names = ['Intercept', 'age', 'sex', 'bmi', 'children', 'smoker']
n, p = X_with_intercept.shape

print(f"\nn = {n} observations")
print(f"p = {p} parameters (including intercept)")

# Run Gibbs sampling
print("\nRunning Gibbs sampler (50,000 iterations, 10,000 warmup, 3 chains)...")
chains = gibbs_lm(
    y=y,
    X=X_with_intercept,
    n_iter=50000,
    warmup=10000,
    n_chains=3,
    seed=42
)

# Extract samples
beta_chains = [chain['beta'] for chain in chains]
sigma2_chains = [chain['sigma2'] for chain in chains]
beta_combined = np.vstack(beta_chains)
sigma2_combined = np.concatenate(sigma2_chains)

# Compute summaries
print("\n" + "="*80)
print("POSTERIOR SUMMARIES")
print("="*80)
print(f"\n{'Parameter':<15} {'Mean':>10} {'Median':>10} {'SD':>10} {'2.5%':>10} {'97.5%':>10}")
print("-"*80)

for i, name in enumerate(feature_names):
    mean_val = np.mean(beta_combined[:, i])
    median_val = np.median(beta_combined[:, i])
    sd_val = np.std(beta_combined[:, i])
    ci_lower = np.percentile(beta_combined[:, i], 2.5)
    ci_upper = np.percentile(beta_combined[:, i], 97.5)
    print(f"{name:<15} {mean_val:>10.4f} {median_val:>10.4f} {sd_val:>10.4f} "
          f"{ci_lower:>10.4f} {ci_upper:>10.4f}")

sigma2_mean = np.mean(sigma2_combined)
sigma2_median = np.median(sigma2_combined)
sigma2_sd = np.std(sigma2_combined)
sigma2_ci_lower = np.percentile(sigma2_combined, 2.5)
sigma2_ci_upper = np.percentile(sigma2_combined, 97.5)

print(f"{'σ²':<15} {sigma2_mean:>10.4f} {sigma2_median:>10.4f} {sigma2_sd:>10.4f} "
      f"{sigma2_ci_lower:>10.4f} {sigma2_ci_upper:>10.4f}")

print("-"*80)

# Compute R-hat for convergence
def compute_rhat(chains_list):
    """Compute Gelman-Rubin R-hat statistic"""
    m = len(chains_list)
    n = len(chains_list[0])
    
    chain_means = np.array([np.mean(chain) for chain in chains_list])
    chain_vars = np.array([np.var(chain, ddof=1) for chain in chains_list])
    
    overall_mean = np.mean(chain_means)
    between_variance = n * np.var(chain_means, ddof=1)
    within_variance = np.mean(chain_vars)
    
    var_plus = ((n - 1) / n) * within_variance + (1 / n) * between_variance
    rhat = np.sqrt(var_plus / within_variance)
    
    return rhat

print("\n" + "="*80)
print("CONVERGENCE DIAGNOSTICS (R-hat)")
print("="*80)
print(f"\n{'Parameter':<15} {'R-hat':>10} {'Converged':>12}")
print("-"*80)

converged_all = True
for i, name in enumerate(feature_names):
    chains_i = [chain[:, i] for chain in beta_chains]
    rhat = compute_rhat(chains_i)
    converged = "✓" if rhat < 1.1 else "✗"
    if rhat >= 1.1:
        converged_all = False
    print(f"{name:<15} {rhat:>10.4f} {converged:>12}")

rhat_sigma2 = compute_rhat(sigma2_chains)
converged = "✓" if rhat_sigma2 < 1.1 else "✗"
if rhat_sigma2 >= 1.1:
    converged_all = False
print(f"{'σ²':<15} {rhat_sigma2:>10.4f} {converged:>12}")
print("-"*80)

if converged_all:
    print("\n✓ All parameters have converged (R-hat < 1.1)")
else:
    print("\n✗ Warning: Some parameters have not converged")

# Save results
output_dir = Path('../outputs/final_baseline')
output_dir.mkdir(parents=True, exist_ok=True)

results_df = pd.DataFrame({
    'Parameter': feature_names + ['σ²'],
    'Mean': [np.mean(beta_combined[:, i]) for i in range(len(feature_names))] + [sigma2_mean],
    'Median': [np.median(beta_combined[:, i]) for i in range(len(feature_names))] + [sigma2_median],
    'SD': [np.std(beta_combined[:, i]) for i in range(len(feature_names))] + [sigma2_sd],
    'CI_2.5': [np.percentile(beta_combined[:, i], 2.5) for i in range(len(feature_names))] + [sigma2_ci_lower],
    'CI_97.5': [np.percentile(beta_combined[:, i], 97.5) for i in range(len(feature_names))] + [sigma2_ci_upper]
})

results_df.to_csv(output_dir / 'posterior_estimates.csv', index=False)
print(f"\n✓ Results saved to {output_dir / 'posterior_estimates.csv'}")

# Model fit statistics
y_pred = X_with_intercept @ np.mean(beta_combined, axis=0)
residuals = y - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)

print("\n" + "="*80)
print("MODEL FIT")
print("="*80)
print(f"R² = {r_squared:.4f}")
print(f"RMSE = {np.sqrt(np.mean(residuals**2)):.2f}")
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
