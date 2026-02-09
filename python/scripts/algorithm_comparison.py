import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from scipy import stats
import arviz as az

# Import manual MCMC diagnostics (demonstrating understanding of mechanics)
from mcmc_diagnostics import rhat, ess_geyer


def compare_algorithms(gibbs_results, mh_results, param_names=None, output_dir='../outputs/algorithm_comparison'):
    """
    Comprehensive comparison of Gibbs Sampling vs Metropolis-Hastings.
    
    Compares:
    1. Effective Sample Size (ESS)
    2. Convergence diagnostics (R-hat)
    3. Computational efficiency (ESS per second)
    4. Acceptance rates (for MH)
    5. Autocorrelation
    6. Posterior means and credible intervals
    
    Parameters:
    -----------
    gibbs_results : list of dicts
        Results from Gibbs sampler
    mh_results : list of dicts
        Results from Metropolis-Hastings sampler
    param_names : list of str
        Names of parameters for reporting
    output_dir : str
        Directory to save comparison outputs
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract beta samples
    gibbs_beta = [chain['beta'] for chain in gibbs_results]
    mh_beta = [chain['beta'] for chain in mh_results]
    
    gibbs_sigma2 = [chain['sigma2'] for chain in gibbs_results]
    mh_sigma2 = [chain['sigma2'] for chain in mh_results]
    
    n_chains = len(gibbs_beta)
    n_params = gibbs_beta[0].shape[1]
    
    if param_names is None:
        param_names = [f'β{i}' for i in range(n_params)]
    
    # Get timing information
    gibbs_time = sum(chain.get('time_elapsed', 0) for chain in gibbs_results)
    mh_time = sum(chain.get('time_elapsed', 0) for chain in mh_results)
    
    # ========================================
    # 1. Effective Sample Size (ESS) Comparison
    # ========================================
    print("=" * 70)
    print("ALGORITHM COMPARISON: Gibbs Sampling vs Metropolis-Hastings")
    print("=" * 70)
    
    ess_comparison = []
    
    for j in range(n_params):
        # Gibbs ESS - using manual implementation
        gibbs_param = np.array([chain[:, j] for chain in gibbs_beta])
        gibbs_ess = ess_geyer(gibbs_param)
        
        # MH ESS - using manual implementation
        mh_param = np.array([chain[:, j] for chain in mh_beta])
        mh_ess = ess_geyer(mh_param)
        
        ess_comparison.append({
            'Parameter': param_names[j],
            'Gibbs_ESS': gibbs_ess,
            'MH_ESS': mh_ess,
            'ESS_Ratio': gibbs_ess / mh_ess if mh_ess > 0 else np.inf
        })
    
    # Add sigma2
    gibbs_sigma2_arr = np.array(gibbs_sigma2)
    mh_sigma2_arr = np.array(mh_sigma2)
    gibbs_ess_sigma = ess_geyer(gibbs_sigma2_arr)
    mh_ess_sigma = ess_geyer(mh_sigma2_arr)
    
    ess_comparison.append({
        'Parameter': 'σ²',
        'Gibbs_ESS': gibbs_ess_sigma,
        'MH_ESS': mh_ess_sigma,
        'ESS_Ratio': gibbs_ess_sigma / mh_ess_sigma if mh_ess_sigma > 0 else np.inf
    })
    
    ess_df = pd.DataFrame(ess_comparison)
    
    print("\n1. EFFECTIVE SAMPLE SIZE (ESS)")
    print("-" * 70)
    print(ess_df.to_string(index=False))
    print(f"\nAverage ESS Ratio (Gibbs/MH): {ess_df['ESS_Ratio'].mean():.3f}")
    
    # Save ESS comparison
    ess_df.to_csv(output_path / 'ess_comparison.csv', index=False)
    
    # ========================================
    # 2. Computational Efficiency (ESS per second)
    # ========================================
    print("\n2. COMPUTATIONAL EFFICIENCY")
    print("-" * 70)
    
    total_gibbs_samples = sum(chain.shape[0] for chain in gibbs_beta)
    total_mh_samples = sum(chain.shape[0] for chain in mh_beta)
    
    avg_gibbs_ess = ess_df[ess_df['Parameter'] != 'σ²']['Gibbs_ESS'].mean()
    avg_mh_ess = ess_df[ess_df['Parameter'] != 'σ²']['MH_ESS'].mean()
    
    gibbs_ess_per_sec = avg_gibbs_ess / gibbs_time if gibbs_time > 0 else 0
    mh_ess_per_sec = avg_mh_ess / mh_time if mh_time > 0 else 0
    
    efficiency_data = {
        'Algorithm': ['Gibbs', 'Metropolis-Hastings'],
        'Total_Time_s': [gibbs_time, mh_time],
        'Total_Samples': [total_gibbs_samples, total_mh_samples],
        'Avg_ESS': [avg_gibbs_ess, avg_mh_ess],
        'ESS_per_Second': [gibbs_ess_per_sec, mh_ess_per_sec],
        'Samples_per_Second': [total_gibbs_samples / gibbs_time if gibbs_time > 0 else 0,
                               total_mh_samples / mh_time if mh_time > 0 else 0]
    }
    
    efficiency_df = pd.DataFrame(efficiency_data)
    print(efficiency_df.to_string(index=False))
    print(f"\nEfficiency Ratio (Gibbs/MH): {gibbs_ess_per_sec / mh_ess_per_sec:.3f}x")
    
    efficiency_df.to_csv(output_path / 'efficiency_comparison.csv', index=False)
    
    # ========================================
    # 3. Convergence Diagnostics (R-hat)
    # ========================================
    print("\n3. CONVERGENCE DIAGNOSTICS (R-hat)")
    print("-" * 70)
    
    rhat_comparison = []
    
    for j in range(n_params):
        gibbs_param = np.array([chain[:, j] for chain in gibbs_beta])
        mh_param = np.array([chain[:, j] for chain in mh_beta])
        
        # Using manual R-hat implementation
        gibbs_rhat = rhat(gibbs_param)
        mh_rhat = rhat(mh_param)
        
        rhat_comparison.append({
            'Parameter': param_names[j],
            'Gibbs_Rhat': gibbs_rhat,
            'MH_Rhat': mh_rhat,
            'Both_Converged': (gibbs_rhat < 1.01) and (mh_rhat < 1.01)
        })
    
    # Add sigma2
    gibbs_rhat_sigma = rhat(gibbs_sigma2_arr)
    mh_rhat_sigma = rhat(mh_sigma2_arr)
    
    rhat_comparison.append({
        'Parameter': 'σ²',
        'Gibbs_Rhat': gibbs_rhat_sigma,
        'MH_Rhat': mh_rhat_sigma,
        'Both_Converged': (gibbs_rhat_sigma < 1.01) and (mh_rhat_sigma < 1.01)
    })
    
    rhat_df = pd.DataFrame(rhat_comparison)
    print(rhat_df.to_string(index=False))
    print(f"\nAll parameters converged: {rhat_df['Both_Converged'].all()}")
    
    rhat_df.to_csv(output_path / 'rhat_comparison.csv', index=False)
    
    # ========================================
    # 4. Acceptance Rates (MH only)
    # ========================================
    if 'acceptance_rate_beta' in mh_results[0]:
        print("\n4. ACCEPTANCE RATES (Metropolis-Hastings)")
        print("-" * 70)
        
        accept_rates = []
        for i, chain in enumerate(mh_results):
            accept_rates.append({
                'Chain': i + 1,
                'Beta_Accept_Rate': chain.get('acceptance_rate_beta', 0),
                'Sigma2_Accept_Rate': chain.get('acceptance_rate_sigma2', 0)
            })
        
        accept_df = pd.DataFrame(accept_rates)
        print(accept_df.to_string(index=False))
        print(f"\nAverage Beta acceptance: {accept_df['Beta_Accept_Rate'].mean():.3f}")
        print(f"Average Sigma2 acceptance: {accept_df['Sigma2_Accept_Rate'].mean():.3f}")
        print("Note: Optimal acceptance rate for RW-MH is ~0.234 (high-dim) to 0.44 (1-dim)")
        
        accept_df.to_csv(output_path / 'mh_acceptance_rates.csv', index=False)
    
    # ========================================
    # 5. Posterior Estimates Comparison
    # ========================================
    print("\n5. POSTERIOR ESTIMATES")
    print("-" * 70)
    
    estimates = []
    
    for j in range(n_params):
        gibbs_samples = np.concatenate([chain[:, j] for chain in gibbs_beta])
        mh_samples = np.concatenate([chain[:, j] for chain in mh_beta])
        
        estimates.append({
            'Parameter': param_names[j],
            'Gibbs_Mean': np.mean(gibbs_samples),
            'MH_Mean': np.mean(mh_samples),
            'Difference': np.mean(gibbs_samples) - np.mean(mh_samples),
            'Gibbs_SD': np.std(gibbs_samples),
            'MH_SD': np.std(mh_samples)
        })
    
    # Add sigma2
    gibbs_sigma_samples = np.concatenate(gibbs_sigma2)
    mh_sigma_samples = np.concatenate(mh_sigma2)
    
    estimates.append({
        'Parameter': 'σ²',
        'Gibbs_Mean': np.mean(gibbs_sigma_samples),
        'MH_Mean': np.mean(mh_sigma_samples),
        'Difference': np.mean(gibbs_sigma_samples) - np.mean(mh_sigma_samples),
        'Gibbs_SD': np.std(gibbs_sigma_samples),
        'MH_SD': np.std(mh_sigma_samples)
    })
    
    estimates_df = pd.DataFrame(estimates)
    print(estimates_df.to_string(index=False))
    print("\nNote: Small differences indicate both algorithms target the same posterior")
    
    estimates_df.to_csv(output_path / 'posterior_estimates_comparison.csv', index=False)
    
    # ========================================
    # 6. Autocorrelation Comparison
    # ========================================
    print("\n6. AUTOCORRELATION ANALYSIS")
    print("-" * 70)
    
    create_autocorr_comparison_plot(gibbs_beta, mh_beta, param_names, output_path)
    print(f"Autocorrelation plots saved to {output_path}")
    
    # ========================================
    # 7. Summary Statistics
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Gibbs Sampling: {avg_gibbs_ess:.0f} ESS in {gibbs_time:.2f}s ({gibbs_ess_per_sec:.1f} ESS/s)")
    print(f"✓ Metropolis-Hastings: {avg_mh_ess:.0f} ESS in {mh_time:.2f}s ({mh_ess_per_sec:.1f} ESS/s)")
    print(f"✓ Winner: {'Gibbs' if gibbs_ess_per_sec > mh_ess_per_sec else 'MH'} " +
          f"({max(gibbs_ess_per_sec, mh_ess_per_sec) / min(gibbs_ess_per_sec, mh_ess_per_sec):.2f}x faster)")
    print(f"✓ Both algorithms converged: {rhat_df['Both_Converged'].all()}")
    print(f"✓ Posterior estimates agree: {np.abs(estimates_df['Difference']).max() < 0.01}")
    print("=" * 70)
    
    return {
        'ess': ess_df,
        'efficiency': efficiency_df,
        'rhat': rhat_df,
        'estimates': estimates_df
    }


def create_autocorr_comparison_plot(gibbs_chains, mh_chains, param_names, output_dir, max_lag=50):
    """
    Create side-by-side autocorrelation plots comparing Gibbs and MH.
    """
    n_params = min(3, gibbs_chains[0].shape[1])  # Plot first 3 parameters
    
    fig, axes = plt.subplots(n_params, 2, figsize=(14, 4 * n_params))
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    for j in range(n_params):
        # Gibbs autocorrelation
        gibbs_samples = gibbs_chains[0][:, j]  # Use first chain
        gibbs_acf = compute_acf(gibbs_samples, max_lag)
        
        axes[j, 0].bar(range(len(gibbs_acf)), gibbs_acf, color='steelblue', alpha=0.7)
        axes[j, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[j, 0].axhline(y=1.96/np.sqrt(len(gibbs_samples)), color='red', linestyle='--', linewidth=1)
        axes[j, 0].axhline(y=-1.96/np.sqrt(len(gibbs_samples)), color='red', linestyle='--', linewidth=1)
        axes[j, 0].set_xlabel('Lag', fontsize=11)
        axes[j, 0].set_ylabel('ACF', fontsize=11)
        axes[j, 0].set_title(f'Gibbs: {param_names[j]}', fontsize=12, fontweight='bold')
        axes[j, 0].grid(True, alpha=0.3)
        
        # MH autocorrelation
        mh_samples = mh_chains[0][:, j]
        mh_acf = compute_acf(mh_samples, max_lag)
        
        axes[j, 1].bar(range(len(mh_acf)), mh_acf, color='coral', alpha=0.7)
        axes[j, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        axes[j, 1].axhline(y=1.96/np.sqrt(len(mh_samples)), color='red', linestyle='--', linewidth=1)
        axes[j, 1].axhline(y=-1.96/np.sqrt(len(mh_samples)), color='red', linestyle='--', linewidth=1)
        axes[j, 1].set_xlabel('Lag', fontsize=11)
        axes[j, 1].set_ylabel('ACF', fontsize=11)
        axes[j, 1].set_title(f'Metropolis-Hastings: {param_names[j]}', fontsize=12, fontweight='bold')
        axes[j, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'autocorrelation_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def compute_acf(x, max_lag=50):
    """Compute autocorrelation function."""
    x = x - np.mean(x)
    c0 = np.dot(x, x) / len(x)
    acf = np.array([np.dot(x[lag:], x[:-lag]) / len(x) / c0 if lag > 0 else 1.0 
                    for lag in range(max_lag)])
    return acf


def create_comparison_summary_plot(gibbs_results, mh_results, param_names, output_dir):
    """
    Create a comprehensive visual summary comparing the two algorithms.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Extract data
    gibbs_beta = [chain['beta'] for chain in gibbs_results]
    mh_beta = [chain['beta'] for chain in mh_results]
    n_params = gibbs_beta[0].shape[1]
    
    # 1. ESS comparison (bar plot)
    ax1 = fig.add_subplot(gs[0, 0])
    ess_gibbs = [az.ess(np.array([chain[:, j] for chain in gibbs_beta])) for j in range(min(5, n_params))]
    ess_mh = [az.ess(np.array([chain[:, j] for chain in mh_beta])) for j in range(min(5, n_params))]
    
    x = np.arange(len(ess_gibbs))
    width = 0.35
    ax1.bar(x - width/2, ess_gibbs, width, label='Gibbs', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, ess_mh, width, label='MH', color='coral', alpha=0.8)
    ax1.set_xlabel('Parameter Index', fontsize=11)
    ax1.set_ylabel('Effective Sample Size', fontsize=11)
    ax1.set_title('ESS Comparison', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Posterior density comparison (first parameter)
    ax2 = fig.add_subplot(gs[0, 1])
    gibbs_samples = np.concatenate([chain[:, 0] for chain in gibbs_beta])
    mh_samples = np.concatenate([chain[:, 0] for chain in mh_beta])
    
    ax2.hist(gibbs_samples, bins=50, density=True, alpha=0.6, label='Gibbs', color='steelblue')
    ax2.hist(mh_samples, bins=50, density=True, alpha=0.6, label='MH', color='coral')
    ax2.set_xlabel(param_names[0], fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Posterior Distribution Comparison', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Trace plot comparison (first parameter)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(gibbs_beta[0][:1000, 0], color='steelblue', alpha=0.7, label='Gibbs', linewidth=1)
    ax3.plot(mh_beta[0][:1000, 0], color='coral', alpha=0.7, label='MH', linewidth=1)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel(param_names[0], fontsize=11)
    ax3.set_title('Trace Plot (First 1000 iterations)', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.savefig(Path(output_dir) / 'algorithm_comparison_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison summary plot saved to {output_dir}")
