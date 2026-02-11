"""
Residual diagnostics for Bayesian regression.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats


def compute_residuals(y_obs, X, beta_samples, sigma2_samples):
    """Compute residuals using posterior mean predictions."""
    beta_mean = np.mean(beta_samples, axis=0)
    y_pred = X @ beta_mean
    residuals = y_obs - y_pred
    
    return residuals, y_pred


def standardized_residuals(residuals, sigma2_samples, X, leverage=None):
    """Compute standardized (Pearson) residuals."""
    sigma_mean = np.sqrt(np.mean(sigma2_samples))
    
    if leverage is None:
        H = X @ np.linalg.inv(X.T @ X) @ X.T
        leverage = np.diag(H)
    
    std_residuals = residuals / (sigma_mean * np.sqrt(1 - leverage))
    
    return std_residuals


def create_residual_diagnostic_plot(y_obs, X, beta_samples, sigma2_samples, 
                                   model_name, output_dir='../outputs'):
    """
    4-panel residual diagnostic plot:
    1. Residuals vs Fitted
    2. QQ plot
    3. Scale-Location
    4. Histogram of standardized residuals
    """
    residuals, y_pred = compute_residuals(y_obs, X, beta_samples, sigma2_samples)
    
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    leverage = np.diag(H)
    
    std_residuals = standardized_residuals(residuals, sigma2_samples, X, leverage)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(f'Residual Diagnostics: {model_name}', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, alpha=0.5, s=40, edgecolors='k', linewidths=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Zero line')
    
    from scipy.signal import savgol_filter
    sorted_idx = np.argsort(y_pred)
    if len(y_pred) > 50:
        window = min(51, len(y_pred) // 5)
        if window % 2 == 0:
            window += 1
        smooth = savgol_filter(residuals[sorted_idx], window, 3)
        ax.plot(y_pred[sorted_idx], smooth, color='blue', linewidth=2, 
                label='LOWESS smoother')
    
    ax.set_xlabel('Fitted values', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residuals vs Fitted', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    stats.probplot(std_residuals, dist="norm", plot=ax)
    ax.set_title('Normal Q-Q Plot', fontsize=13, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax.set_ylabel('Standardized Residuals', fontsize=12)
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    sqrt_abs_std_res = np.sqrt(np.abs(std_residuals))
    ax.scatter(y_pred, sqrt_abs_std_res, alpha=0.5, s=40, 
               edgecolors='k', linewidths=0.5)
    
    if len(y_pred) > 50:
        smooth = savgol_filter(sqrt_abs_std_res[sorted_idx], window, 3)
        ax.plot(y_pred[sorted_idx], smooth, color='red', linewidth=2, 
                label='LOWESS smoother')
    
    ax.set_xlabel('Fitted values', fontsize=12)
    ax.set_ylabel('√|Standardized residuals|', fontsize=12)
    ax.set_title('Scale-Location', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Panel 4: Histogram
    ax = axes[1, 1]
    ax.hist(std_residuals, bins=30, density=True, alpha=0.7, 
            edgecolor='black', label='Residuals')
    
    # Overlay normal distribution
    x_range = np.linspace(std_residuals.min(), std_residuals.max(), 100)
    ax.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'r-', 
            linewidth=2, label='N(0,1)')
    
    ax.set_xlabel('Standardized Residuals', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Residuals', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Add diagnostics text
    # Shapiro-Wilk test for normality
    shapiro_stat, shapiro_p = stats.shapiro(std_residuals)
    
    # Durbin-Watson (approximate autocorrelation test)
    dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
    
    textstr = f'Shapiro-Wilk: W={shapiro_stat:.4f}, p={shapiro_p:.4e}\n'
    textstr += f'Durbin-Watson: {dw_stat:.4f}\n'
    textstr += f'Mean residual: {np.mean(residuals):.2e}'
    
    fig.text(0.02, 0.02, textstr, fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             verticalalignment='bottom')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / model_name / 'residual_diagnostics.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    
    # Return diagnostics
    return {
        'residuals': residuals,
        'std_residuals': std_residuals,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'dw_stat': dw_stat
    }


def create_residual_comparison_plot(diagnostics_dict, output_dir='../outputs'):
    """
    Create side-by-side comparison of residuals across models.
    
    Parameters:
    -----------
    diagnostics_dict : dict
        Dictionary with model names as keys and diagnostic dicts as values
    """
    n_models = len(diagnostics_dict)
    fig, axes = plt.subplots(2, n_models, figsize=(6*n_models, 10))
    
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Residual Comparison Across Models', 
                 fontsize=16, fontweight='bold')
    
    for idx, (model_name, diag) in enumerate(diagnostics_dict.items()):
        std_resid = diag['std_residuals']
        
        # Row 1: QQ plots
        ax = axes[0, idx]
        stats.probplot(std_resid, dist="norm", plot=ax)
        ax.set_title(f'{model_name}\nShapiro-Wilk p={diag["shapiro_p"]:.3e}', 
                     fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Row 2: Histograms
        ax = axes[1, idx]
        ax.hist(std_resid, bins=30, density=True, alpha=0.7, 
                edgecolor='black')
        x_range = np.linspace(std_resid.min(), std_resid.max(), 100)
        ax.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'r-', linewidth=2)
        ax.set_xlabel('Standardized Residuals', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'residual_comparison.png'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("Residual diagnostics module loaded.")
    print("Use create_residual_diagnostic_plot() to generate diagnostics.")
