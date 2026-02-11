#!/usr/bin/env python
"""
Complete analysis runner.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys

from data_preprocessing import preprocess_data
from model_setup import create_design_matrix, run_ols_baseline
from gibbs_sampling import gibbs_lm, beta_trace_plot, sigma2_trace_plot
from convergence_detection import acf_plot_beta, acf_plot_sigma2
from posterior_inference import posterior_predictive, ppc_plot, PPC_density_overlay, ppc_residual_plot
from mcmc_diagnostics import rhat, ess_geyer

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

print("="*80)
print("Bayesian MCMC Insurance Regression - Full Analysis")
print("="*80)

print("\n[1/7] Loading data...")

df = pd.read_csv('../../data/expenses_cleaned.csv')
print(f"   Data shape: {df.shape}")
print(f"   Columns: {list(df.columns)}")

y = df['charges'].values
X = df.drop('charges', axis=1).values
n, p = X.shape

X_with_intercept = np.column_stack([np.ones(n), X])
feature_names = ['Intercept'] + list(df.drop('charges', axis=1).columns)

print(f"   n = {n} observations")
print(f"   p = {p} predictors (+ intercept)")
print(f"   Features: {feature_names}")

print("\n[2/7] Running Gibbs sampling...")
print("   Parameters: 50,000 iterations, 10,000 warmup, 3 chains")

# Run Gibbs sampler
chains_baseline = gibbs_lm(
    y=y,
    X=X_with_intercept,
    n_iter=50000,
    warmup=10000,
    n_chains=3,
    seed=42
)

# Extract samples
beta_chains = [chain['beta'] for chain in chains_baseline]
sigma2_chains = [chain['sigma2'] for chain in chains_baseline]

print("   ✓ Sampling complete")

# ============================================================================
# 3. CONVERGENCE DIAGNOSTICS
# ============================================================================
print("\n[3/7] Computing convergence diagnostics...")

# Create output directory
output_dir = Path('../outputs/baseline_model')
output_dir.mkdir(parents=True, exist_ok=True)

# Compute R-hat using manual implementation
rhat_beta = []
for j in range(X_with_intercept.shape[1]):
    chains_j = np.array([chain[:, j] for chain in beta_chains])  # shape: (n_chains, n_samples)
    rhat_beta.append(rhat(chains_j))

rhat_sigma2 = rhat(np.array(sigma2_chains))

print(f"   R-hat (beta): {[f'{r:.4f}' for r in rhat_beta]}")
print(f"   R-hat (sigma2): {rhat_sigma2:.4f}")

# Check convergence
converged = all(r < 1.1 for r in rhat_beta) and rhat_sigma2 < 1.1
print(f"   Convergence: {'✓ CONVERGED' if converged else '✗ NOT CONVERGED'}")

# Compute ESS
ess_dir = output_dir / 'ESS_tables'
ess_dir.mkdir(exist_ok=True)

ess_beta = []
for j in range(X_with_intercept.shape[1]):
    chains_j = np.array([chain[:, j] for chain in beta_chains])  # shape: (n_chains, n_samples)
    ess_beta.append(ess_geyer(chains_j))

# Save ESS table
ess_df = pd.DataFrame({
    'Parameter': feature_names,
    'ESS': ess_beta
})
ess_df.to_csv(ess_dir / 'ESS_beta.txt', index=False, sep='\t')
print(f"   ESS (beta): {[f'{e:.0f}' for e in ess_beta]}")

# ============================================================================
# 4. POSTERIOR SUMMARIES
# ============================================================================
print("\n[4/7] Computing posterior summaries...")

# Combine chains
beta_combined = np.vstack(beta_chains)
sigma2_combined = np.concatenate(sigma2_chains)

# Compute summaries manually
mean_beta = np.mean(beta_combined, axis=0)
median_beta = np.median(beta_combined, axis=0)
sd_beta = np.std(beta_combined, axis=0, ddof=1)
ci_lower = np.percentile(beta_combined, 2.5, axis=0)
ci_upper = np.percentile(beta_combined, 97.5, axis=0)

print("\n   Posterior Estimates:")
print("   " + "-"*70)
print(f"   {'Parameter':<20} {'Mean':>10} {'Median':>10} {'SD':>10}")
print("   " + "-"*70)
for i, name in enumerate(feature_names):
    print(f"   {name:<20} {mean_beta[i]:>10.4f} "
          f"{median_beta[i]:>10.4f} {sd_beta[i]:>10.4f}")
print("   " + "-"*70)

# Save posterior estimates
posterior_df = pd.DataFrame({
    'Parameter': feature_names,
    'Mean': mean_beta,
    'Median': median_beta,
    'SD': sd_beta,
    'CI_lower': ci_lower,
    'CI_upper': ci_upper
})
posterior_df.to_csv(output_dir / 'posterior_estimates.csv', index=False)

# ============================================================================
# 5. GENERATE TRACE PLOTS
# ============================================================================
print("\n[5/7] Generating trace plots...")

# Note: beta_trace_plot expects list of arrays, not feature_names parameter
beta_trace_plot(beta_chains, model_name='baseline_model', plot_dir='../outputs')
sigma2_trace_plot(sigma2_chains, model_name='baseline_model', plot_dir='../outputs')

print("   ✓ Trace plots saved")

# ============================================================================
# 6. RESIDUAL DIAGNOSTICS
# ============================================================================
print("\n[6/7] Computing residual diagnostics...")

# Use posterior mean for predictions
y_pred = X_with_intercept @ mean_beta
residuals = y - y_pred

# Simple residual plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residuals vs fitted
axes[0].scatter(y_pred, residuals, alpha=0.5)
axes[0].axhline(y=0, color='r', linestyle='--')
axes[0].set_xlabel('Fitted values')
axes[0].set_ylabel('Residuals')
axes[0].set_title('Residuals vs Fitted')

# Q-Q plot
from scipy import stats as sp_stats
sp_stats.probplot(residuals, dist="norm", plot=axes[1])
axes[1].set_title('Normal Q-Q Plot')

plt.tight_layout()
plt.savefig(output_dir / 'residual_diagnostics.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ Residual diagnostics saved")

# ============================================================================
# 7. POSTERIOR PREDICTIVE CHECKS
# ============================================================================
print("\n[7/7] Posterior predictive checks...")

# Sample from posterior predictive
n_samples = 1000
n_obs = len(y)
y_rep = np.zeros((n_samples, n_obs))

# Random sample from posterior
idx = np.random.choice(len(beta_combined), size=n_samples, replace=False)

for i, j in enumerate(idx):
    beta_sample = beta_combined[j]
    sigma2_sample = sigma2_combined[j]
    y_rep[i] = X_with_intercept @ beta_sample + \
               np.random.normal(0, np.sqrt(sigma2_sample), n_obs)

# Compute test statistics
T_obs = np.mean(y)
T_rep = np.mean(y_rep, axis=1)
p_value = np.mean(T_rep >= T_obs)

print(f"   Posterior predictive p-value (mean): {p_value:.4f}")

# Plot PPC
ppc_dir = output_dir / 'PPC'
ppc_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(T_rep, bins=50, alpha=0.7, edgecolor='black', label='Replicated data')
ax.axvline(T_obs, color='red', linestyle='--', linewidth=2, label='Observed data')
ax.set_xlabel('Mean charges', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Posterior Predictive Check: Mean', fontsize=14)
ax.legend()
plt.tight_layout()
plt.savefig(ppc_dir / 'ppc_mean.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✓ PPC plots saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nResults saved to: {output_dir.absolute()}")
print("\nKey findings:")
print(f"  - All chains converged (R-hat < 1.1): {converged}")
print(f"  - Effective sample sizes: {int(np.mean(ess_beta)):.0f} (average)")
print(f"  - Strongest predictor: {feature_names[np.argmax(np.abs(mean_beta))]} "
      f"(β = {mean_beta[np.argmax(np.abs(mean_beta))]:.3f})")
print(f"  - Model R²: {1 - np.var(residuals)/np.var(y):.4f}")
print("\n" + "="*80)
