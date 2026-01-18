"""
Posterior Inference and Predictive Checks

This module provides functions for:
1. Posterior Predictive Distribution
2. Posterior Predictive Checks (PPC)
   - Scatter plots comparing observed vs predicted
   - Density overlays
   - Residual plots

These tools help assess model fit and predictive performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns


def posterior_predictive(beta_list, sigma2_list, X_new):
    """
    Generate posterior predictive draws for new or observed data.
    
    For each posterior sample of (beta, sigma^2), generates a prediction:
    y_rep ~ N(X_new * beta, sigma^2)
    
    Parameters
    ----------
    beta_list : list of np.ndarray
        List of beta samples from each chain (each of shape [n_iter, p])
    sigma2_list : list of np.ndarray
        List of sigma^2 samples from each chain (each of shape [n_iter])
    X_new : np.ndarray
        Design matrix for predictions (n_new x p)
    
    Returns
    -------
    np.ndarray
        Matrix of posterior predictive draws (n_draws x n_new)
        where n_draws is the total number of posterior samples
    """
    # Combine chains into a single posterior sample
    beta_all = np.vstack(beta_list)
    sigma2_all = np.concatenate(sigma2_list)
    
    n_draws = beta_all.shape[0]
    n_new = X_new.shape[0]
    
    # Matrix to store replicated outcomes
    y_rep = np.zeros((n_draws, n_new))
    
    for i in range(n_draws):
        # Linear predictor (mean)
        mu = X_new @ beta_all[i, :]
        
        # Posterior predictive draw
        y_rep[i, :] = np.random.normal(mu, np.sqrt(sigma2_all[i]))
    
    return y_rep


def ppc_plot(y_obs, y_rep, model_name, plot_dir='../../plots'):
    """
    Create posterior predictive check scatter plot.
    
    Compares observed responses with posterior predictive means.
    Points close to the diagonal line indicate good model fit.
    
    Parameters
    ----------
    y_obs : np.ndarray
        Observed response vector
    y_rep : np.ndarray
        Posterior predictive matrix from posterior_predictive()
    model_name : str
        Name of the model (used for folder naming)
    plot_dir : str
        Base directory for saving plots
    """
    # Create output directory
    ppc_dir = Path(plot_dir) / model_name / 'PPC'
    ppc_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute posterior predictive means
    predicted = y_rep.mean(axis=0)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y_obs, predicted, alpha=0.4, s=50, edgecolors='k', linewidth=0.5)
    
    # Add diagonal reference line
    min_val = min(y_obs.min(), predicted.min())
    max_val = max(y_obs.max(), predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('Observed y', fontsize=13)
    ax.set_ylabel('Posterior mean prediction', fontsize=13)
    ax.set_title('Posterior Predictive Check', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    filename = ppc_dir / 'PPC_Scatter.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def PPC_density_overlay(y_obs, y_rep, model_name, n_samples=200, plot_dir='../../plots'):
    """
    Create density overlay plot for posterior predictive checks.
    
    Overlays the density of observed data with densities from replicated datasets.
    If the model fits well, the observed density should be similar to the replicated densities.
    
    Parameters
    ----------
    y_obs : np.ndarray
        Observed response vector
    y_rep : np.ndarray
        Posterior predictive matrix from posterior_predictive()
    model_name : str
        Name of the model (used for folder naming)
    n_samples : int
        Number of replicated datasets to sample for visualization (default: 200)
    plot_dir : str
        Base directory for saving plots
    """
    # Create output directory
    ppc_dir = Path(plot_dir) / model_name / 'PPC'
    ppc_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample replicated datasets
    sample_indices = np.random.choice(y_rep.shape[0], size=n_samples, replace=False)
    y_rep_sample = y_rep[sample_indices, :].flatten()
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot replicated density
    ax.hist(y_rep_sample, bins=50, density=True, alpha=0.4, 
            color='skyblue', label='Replicated', edgecolor='black', linewidth=0.5)
    
    # Plot observed density
    ax.hist(y_obs, bins=50, density=True, alpha=0.4, 
            color='salmon', label='Observed', edgecolor='black', linewidth=0.5)
    
    # Add KDE for smoother visualization
    from scipy.stats import gaussian_kde
    
    # KDE for observed
    kde_obs = gaussian_kde(y_obs)
    x_range = np.linspace(min(y_obs.min(), y_rep_sample.min()),
                          max(y_obs.max(), y_rep_sample.max()), 300)
    ax.plot(x_range, kde_obs(x_range), 'r-', linewidth=2, label='Observed (KDE)')
    
    # KDE for replicated
    kde_rep = gaussian_kde(y_rep_sample)
    ax.plot(x_range, kde_rep(x_range), 'b-', linewidth=2, label='Replicated (KDE)')
    
    ax.set_xlabel('y', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title('Posterior Predictive Density Overlay', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Save plot
    filename = ppc_dir / 'PPC_Density_Overlay.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def ppc_residual_plot(y_obs, y_rep, model_name, plot_dir='../../plots'):
    """
    Create residual plot for posterior predictive checks.
    
    Plots residuals (observed - predicted) against posterior mean predictions.
    Good fit shows randomly scattered residuals around zero with no patterns.
    
    Parameters
    ----------
    y_obs : np.ndarray
        Observed response vector
    y_rep : np.ndarray
        Posterior predictive matrix from posterior_predictive()
    model_name : str
        Name of the model (used for folder naming)
    plot_dir : str
        Base directory for saving plots
    """
    # Create output directory
    ppc_dir = Path(plot_dir) / model_name / 'PPC'
    ppc_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute posterior predictive means and residuals
    predict_mean = y_rep.mean(axis=0)
    residuals = y_obs - predict_mean
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(predict_mean, residuals, alpha=0.4, s=50, 
              edgecolors='k', linewidth=0.5, color='steelblue')
    
    # Add horizontal line at zero
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Posterior Mean Prediction', fontsize=13)
    ax.set_ylabel('Predictive Residual', fontsize=13)
    ax.set_title('Posterior Predictive Residual Check', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Save plot
    filename = ppc_dir / 'PPC_Residuals.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


if __name__ == "__main__":
    print("=" * 60)
    print("POSTERIOR INFERENCE AND PREDICTIVE CHECKS")
    print("=" * 60)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    n = 100
    p = 3
    
    X = np.column_stack([np.ones(n), np.random.randn(n, p-1)])
    true_beta = np.array([2.0, 1.5, -1.0])
    true_sigma2 = 1.0
    
    y_obs = X @ true_beta + np.random.randn(n) * np.sqrt(true_sigma2)
    
    print(f"\nTest data generated:")
    print(f"  n = {n}, p = {p}")
    
    # Generate fake posterior samples
    n_draws = 500
    beta_all = np.random.randn(n_draws, p) * 0.1 + true_beta
    sigma2_all = np.abs(np.random.randn(n_draws) * 0.2 + true_sigma2)
    
    # Package as lists (as if from multiple chains)
    beta_list = [beta_all]
    sigma2_list = [sigma2_all]
    
    print("\nGenerating posterior predictive samples...")
    y_rep = posterior_predictive(beta_list, sigma2_list, X)
    print(f"  y_rep shape: {y_rep.shape}")
    
    print("\nCreating PPC plots...")
    ppc_plot(y_obs, y_rep, 'test_model')
    PPC_density_overlay(y_obs, y_rep, 'test_model')
    ppc_residual_plot(y_obs, y_rep, 'test_model')
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
