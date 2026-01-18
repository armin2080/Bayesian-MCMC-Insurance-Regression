"""
Gibbs Sampler for Bayesian Linear Regression

This module implements a Gibbs sampler for Bayesian linear regression with
conjugate priors. The sampler alternates between sampling regression coefficients
(beta) and error variance (sigma^2).

Prior Distributions:
- beta | sigma^2 ~ N(b0, sigma^2 * B0^(-1))
- sigma^2 ~ Inverse-Gamma(a0, d0)

Posterior Distributions:
- beta | sigma^2, y ~ N(bn, sigma^2 * Bn^(-1))
- sigma^2 | beta, y ~ Inverse-Gamma(an, dn)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import pandas as pd


def gibbs_lm(y, X, n_iter=10000, warmup=2000, n_chains=3,
             b0=None, B0=None, a0=0.01, d0=0.01, seed=123):
    """
    Gibbs sampler for Bayesian linear regression.
    
    Parameters
    ----------
    y : np.ndarray
        Response variable (n x 1)
    X : np.ndarray
        Design matrix (n x p) including intercept
    n_iter : int
        Total number of Gibbs sampling iterations (default: 10000)
    warmup : int
        Number of burn-in samples to discard (default: 2000)
    n_chains : int
        Number of independent MCMC chains (default: 3)
    b0 : np.ndarray, optional
        Prior mean for beta (p x 1). Default is zero vector
    B0 : np.ndarray, optional
        Prior precision matrix for beta (p x p). Default is weak prior (1e-4 * I)
    a0 : float
        Shape parameter for Inverse-Gamma prior on sigma^2 (default: 0.01)
    d0 : float
        Scale parameter for Inverse-Gamma prior on sigma^2 (default: 0.01)
    seed : int
        Random seed base for reproducibility (default: 123)
    
    Returns
    -------
    list of dict
        List of length n_chains, each containing:
        - 'beta': np.ndarray of shape (n_iter - warmup, p) - posterior samples of beta
        - 'sigma2': np.ndarray of shape (n_iter - warmup,) - posterior samples of sigma^2
    """
    
    n = len(y)  # number of observations
    p = X.shape[1]  # number of predictors
    
    # Default priors
    if b0 is None:
        b0 = np.zeros(p)
    if B0 is None:
        B0 = np.eye(p) * 1e-4
    
    # Initialize chain results
    chain_results = []
    
    # Pre-compute for efficiency
    XtX = X.T @ X
    Xty = X.T @ y
    
    for chain in range(n_chains):
        # Set seed for reproducibility
        np.random.seed(seed + chain)
        
        # Storage for samples
        beta_store = np.zeros((n_iter, p))
        sigma2_store = np.zeros(n_iter)
        
        # Initialize parameters
        beta_draw = np.zeros(p)
        sigma2_draw = 1.0
        
        # Gibbs sampling loop
        for iter_idx in range(n_iter):
            
            # ----- Sample beta given sigma^2 -----
            Bn = XtX + B0
            bn = np.linalg.solve(Bn, Xty + B0 @ b0)
            
            # Sample from multivariate normal
            Bn_inv = np.linalg.inv(Bn)
            beta_draw = np.random.multivariate_normal(
                mean=bn,
                cov=sigma2_draw * Bn_inv
            )
            
            # ----- Sample sigma^2 given beta -----
            resid = y - X @ beta_draw
            an = a0 + n / 2
            dn = d0 + 0.5 * np.sum(resid**2)
            
            # Sample from Inverse-Gamma (via Gamma)
            sigma2_draw = 1.0 / np.random.gamma(shape=an, scale=1.0/dn)
            
            # Store samples
            beta_store[iter_idx, :] = beta_draw
            sigma2_store[iter_idx] = sigma2_draw
        
        # Store results after burn-in
        chain_results.append({
            'beta': beta_store[warmup:, :],
            'sigma2': sigma2_store[warmup:]
        })
        
        print(f"Chain {chain + 1} completed.")
    
    return chain_results


def beta_trace_plot(beta_list, model_name, plot_dir='../../plots'):
    """
    Create trace plots for regression coefficients (beta) from multiple chains.
    
    Parameters
    ----------
    beta_list : list of np.ndarray
        List of beta samples from each chain
    model_name : str
        Name of the model (used for folder naming)
    plot_dir : str
        Base directory for saving plots
    """
    n_chains = len(beta_list)
    p = beta_list[0].shape[1]
    
    # Create output directory
    output_dir = Path(plot_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_chains))
    
    for j in range(p):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each chain
        for chain_idx in range(n_chains):
            iterations = np.arange(1, len(beta_list[chain_idx][:, j]) + 1)
            ax.plot(iterations, beta_list[chain_idx][:, j], 
                   color=colors[chain_idx], alpha=0.7, 
                   label=f'Chain {chain_idx + 1}', linewidth=0.5)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel(rf'$\beta_{{{j}}}$', fontsize=12)
        ax.set_title(rf'Trace plot: $\beta_{{{j}}}$', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Save plot
        filename = output_dir / f'beta_trace_{j+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


def sigma2_trace_plot(sigma2_list, model_name, plot_dir='../../plots'):
    """
    Create trace plot for error variance (sigma^2) from multiple chains.
    
    Parameters
    ----------
    sigma2_list : list of np.ndarray
        List of sigma^2 samples from each chain
    model_name : str
        Name of the model (used for folder naming)
    plot_dir : str
        Base directory for saving plots
    """
    n_chains = len(sigma2_list)
    
    # Create output directory
    output_dir = Path(plot_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, n_chains))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot each chain
    for chain_idx in range(n_chains):
        iterations = np.arange(1, len(sigma2_list[chain_idx]) + 1)
        ax.plot(iterations, sigma2_list[chain_idx], 
               color=colors[chain_idx], alpha=0.7,
               label=f'Chain {chain_idx + 1}', linewidth=1.5)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel(r'$\sigma^2$', fontsize=12)
    ax.set_title(r'Trace plot: $\sigma^2$', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Save plot
    filename = output_dir / 'Sigma_trace.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def beta_summary_stats(beta_list):
    """
    Compute posterior summary statistics for regression coefficients.
    
    Parameters
    ----------
    beta_list : list of np.ndarray
        List of beta samples from each chain
    
    Returns
    -------
    pd.DataFrame
        Summary statistics (mean, std, 2.5%, 97.5%) for each beta coefficient
    """
    # Combine all chains
    beta_all = np.vstack(beta_list)
    
    summary = {
        'Mean': np.mean(beta_all, axis=0),
        'SD': np.std(beta_all, axis=0, ddof=1),
        '2.5%': np.percentile(beta_all, 2.5, axis=0),
        '97.5%': np.percentile(beta_all, 97.5, axis=0)
    }
    
    df_summary = pd.DataFrame(summary)
    df_summary = df_summary.round(4)
    
    print("\nBeta Summary Statistics:")
    print(df_summary)
    
    return df_summary


def sigma2_summary_stats(sigma2_list):
    """
    Compute posterior summary statistics for error variance.
    
    Parameters
    ----------
    sigma2_list : list of np.ndarray
        List of sigma^2 samples from each chain
    
    Returns
    -------
    dict
        Summary statistics (mean, std, 2.5%, 97.5%) for sigma^2
    """
    # Combine all chains
    sigma2_all = np.concatenate(sigma2_list)
    
    summary = {
        'Mean': np.mean(sigma2_all),
        'SD': np.std(sigma2_all, ddof=1),
        '2.5%': np.percentile(sigma2_all, 2.5),
        '97.5%': np.percentile(sigma2_all, 97.5)
    }
    
    # Round values
    summary = {k: round(v, 4) for k, v in summary.items()}
    
    print("\nSigma^2 Summary Statistics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return summary


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("GIBBS SAMPLER FOR BAYESIAN LINEAR REGRESSION")
    print("=" * 60)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n = 100
    p = 3
    
    X = np.column_stack([np.ones(n), np.random.randn(n, p-1)])
    true_beta = np.array([2.0, 1.5, -1.0])
    true_sigma2 = 1.0
    
    y = X @ true_beta + np.random.randn(n) * np.sqrt(true_sigma2)
    
    print(f"\nTest data generated:")
    print(f"  n = {n}, p = {p}")
    print(f"  True beta: {true_beta}")
    print(f"  True sigma^2: {true_sigma2}")
    
    # Run Gibbs sampler
    print("\nRunning Gibbs sampler...")
    results = gibbs_lm(y, X, n_iter=5000, warmup=1000, n_chains=3)
    
    # Extract beta and sigma2
    beta_list = [chain['beta'] for chain in results]
    sigma2_list = [chain['sigma2'] for chain in results]
    
    # Summary statistics
    beta_summary = beta_summary_stats(beta_list)
    sigma2_summary = sigma2_summary_stats(sigma2_list)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
