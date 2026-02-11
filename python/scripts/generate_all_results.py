#!/usr/bin/env python
"""
Generate all results for the report.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from data_preprocessing import preprocess_data
from model_setup import create_design_matrix, run_ols_baseline, run_bayesian_model
from gibbs_sampling import gibbs_lm, beta_summary_stats, sigma2_summary_stats
from metropolis_hastings import metropolis_hastings_lm
from algorithm_comparison import compare_algorithms

print("="*80)
print("Comprehensive results generation")
print("="*80)

df = pd.read_csv('../../data/expenses_cleaned.csv')
print(f"\nData loaded: {df.shape}")

print("\n[1/5] Baseline - Gibbs Sampling...")
X, y, feature_names = create_design_matrix(df, formula='baseline')

# Run Gibbs
gibbs_results = gibbs_lm(
    y=y, X=X, 
    n_iter=10000, warmup=2000, n_chains=4, 
    seed=123
)

# Save results
beta_list = [chain['beta'] for chain in gibbs_results]
sigma2_list = [chain['sigma2'] for chain in gibbs_results]

print("Gibbs Sampling - Posterior Summary:")
beta_summary = beta_summary_stats(beta_list)
sigma2_summary = sigma2_summary_stats(sigma2_list)

# Save to CSV
output_dir = Path('../outputs/final_baseline')
output_dir.mkdir(parents=True, exist_ok=True)
beta_summary.to_csv(output_dir / 'posterior_estimates.csv', index=True)

# ============================================================================
# 2. BASELINE MODEL - METROPOLIS-HASTINGS
# ============================================================================
print("\n[2/5] Running Baseline Model - Metropolis-Hastings...")

mh_results = metropolis_hastings_lm(
    y=y, X=X,
    n_iter=10000, warmup=2000, n_chains=4,
    proposal_sd_beta=0.5, proposal_sd_sigma2=0.3,
    seed=123
)

print("MH Sampling - Acceptance Rates:")
for i, chain in enumerate(mh_results):
    print(f"  Chain {i+1}: Beta={chain.get('acceptance_rate_beta', 0):.3f}, "
          f"Sigma2={chain.get('acceptance_rate_sigma2', 0):.3f}")

# ============================================================================
# 3. ALGORITHM COMPARISON
# ============================================================================
print("\n[3/5] Comparing Algorithms (Gibbs vs MH)...")

compare_algorithms(
    gibbs_results=gibbs_results,
    mh_results=mh_results,
    param_names=feature_names,  # Include all parameter names
    output_dir='../outputs/algorithm_comparison'
)

print("Algorithm comparison saved.")

# ============================================================================
# 4. LOG-TRANSFORMED MODEL
# ============================================================================
print("\n[4/5] Running Log-Transformed Model...")
X_log, y_log, feature_names_log = create_design_matrix(df, formula='log')

gibbs_log = gibbs_lm(
    y=y_log, X=X_log,
    n_iter=10000, warmup=2000, n_chains=4,
    seed=456
)

beta_list_log = [chain['beta'] for chain in gibbs_log]
sigma2_list_log = [chain['sigma2'] for chain in gibbs_log]

print("Log Model - Posterior Summary:")
beta_summary_log = beta_summary_stats(beta_list_log)
sigma2_summary_log = sigma2_summary_stats(sigma2_list_log)

output_dir_log = Path('../outputs/log_transformed')
output_dir_log.mkdir(parents=True, exist_ok=True)
beta_summary_log.to_csv(output_dir_log / 'posterior_estimates.csv', index=True)

# ============================================================================
# 5. INTERACTION MODEL
# ============================================================================
print("\n[5/5] Running Interaction Model (smoker:bmi)...")
X_int, y_int, feature_names_int = create_design_matrix(df, formula='interaction')

gibbs_int = gibbs_lm(
    y=y_int, X=X_int,
    n_iter=10000, warmup=2000, n_chains=4,
    seed=789
)

beta_list_int = [chain['beta'] for chain in gibbs_int]
sigma2_list_int = [chain['sigma2'] for chain in gibbs_int]

print("Interaction Model - Posterior Summary:")
beta_summary_int = beta_summary_stats(beta_list_int)
sigma2_summary_int = sigma2_summary_stats(sigma2_list_int)

output_dir_int = Path('../outputs/interaction')
output_dir_int.mkdir(parents=True, exist_ok=True)
beta_summary_int.to_csv(output_dir_int / 'posterior_estimates.csv', index=True)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL RESULTS GENERATED SUCCESSFULLY")
print("="*80)
print("\nOutputs saved to:")
print("  - ../outputs/final_baseline/")
print("  - ../outputs/algorithm_comparison/")
print("  - ../outputs/log_transformed/")
print("  - ../outputs/interaction/")
print("\n" + "="*80)
