import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from statsmodels.tsa.stattools import acf


def acf_plot_beta(beta_list, model_name, lag_max=50, ylim_zoom=0.1, plot_dir='../outputs'):
    acf_dir = Path(plot_dir) / model_name / 'ACF_plots'
    full_dir = acf_dir / 'full'
    zoom_dir = acf_dir / 'zoomed'
    full_dir.mkdir(parents=True, exist_ok=True)
    zoom_dir.mkdir(parents=True, exist_ok=True)
    
    p = beta_list[0].shape[1]
    for j in range(p):
        beta_samples = np.concatenate([chain[:, j] for chain in beta_list])
        acf_values = acf(beta_samples, nlags=lag_max, fft=True)
        lags = np.arange(len(acf_values))
        ci = 1.96 / np.sqrt(len(beta_samples))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stem(lags, acf_values, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
        ax.axhline(y=0, color='red', linestyle='-', linewidth=1)
        ax.axhline(y=ci, color='blue', linestyle='--', linewidth=1)
        ax.axhline(y=-ci, color='blue', linestyle='--', linewidth=1)
        ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylabel('ACF', fontsize=12)
        ax.set_title(rf'Full ACF of $\beta_{{{j}}}$', fontsize=14)
        ax.grid(True, alpha=0.3)
        full_file = full_dir / f'acf_beta_{j}_full.png'
        plt.savefig(full_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {full_file}")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.stem(lags[1:], acf_values[1:], linefmt='C0-', markerfmt='C0o', basefmt='C0-')
        ax.axhline(y=0, color='red', linestyle='-', linewidth=1)
        ax.axhline(y=ci, color='blue', linestyle='--', linewidth=1)
        ax.axhline(y=-ci, color='blue', linestyle='--', linewidth=1)
        ax.set_ylim(-ylim_zoom, ylim_zoom)
        ax.set_xlabel('Lag', fontsize=12)
        ax.set_ylabel('ACF', fontsize=12)
        ax.set_title(rf'Zoomed ACF of $\beta_{{{j}}}$', fontsize=14)
        ax.grid(True, alpha=0.3)
        zoom_file = zoom_dir / f'acf_beta_{j}_zoomed.png'
        plt.savefig(zoom_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {zoom_file}")


def acf_plot_sigma2(sigma2_list, model_name, lag_max=50, ylim_zoom=0.1, plot_dir='../outputs'):
    acf_dir = Path(plot_dir) / model_name / 'ACF_plots'
    full_dir = acf_dir / 'full'
    zoom_dir = acf_dir / 'zoomed'
    
    full_dir.mkdir(parents=True, exist_ok=True)
    zoom_dir.mkdir(parents=True, exist_ok=True)
    
    sigma2_samples = np.concatenate(sigma2_list)
    acf_values = acf(sigma2_samples, nlags=lag_max, fft=True)
    lags = np.arange(len(acf_values))
    
    # full ACF
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stem(lags, acf_values, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
    ax.axhline(y=0, color='red', linestyle='-', linewidth=1)
    
    ci = 1.96 / np.sqrt(len(sigma2_samples))
    ax.axhline(y=ci, color='blue', linestyle='--', linewidth=1)
    ax.axhline(y=-ci, color='blue', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.set_title(r'Full ACF of $\sigma^2$', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    full_file = full_dir / 'acf_sigma2_full.png'
    plt.savefig(full_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {full_file}")
    
    # zoomed ACF
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stem(lags[1:], acf_values[1:], linefmt='C0-', markerfmt='C0o', basefmt='C0-')
    ax.axhline(y=0, color='red', linestyle='-', linewidth=1)
    ax.axhline(y=ci, color='blue', linestyle='--', linewidth=1)
    ax.axhline(y=-ci, color='blue', linestyle='--', linewidth=1)
    
    ax.set_ylim(-ylim_zoom, ylim_zoom)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('ACF', fontsize=12)
    ax.set_title(r'Zoomed ACF of $\sigma^2$', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    zoom_file = zoom_dir / 'acf_sigma2_zoomed.png'
    plt.savefig(zoom_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {zoom_file}")


def effective_sample_size(x, max_lag=100):
    acf_values = acf(x, nlags=max_lag, fft=True)[1:]
    positive_acf = acf_values[acf_values > 0]
    ess = len(x) / (1 + 2 * np.sum(positive_acf))
    return ess


def ess_beta_table(beta_list, X, model_name, output_dir='../../r/outputs'):
    ess_dir = Path(output_dir) / model_name / 'ESS_tables'
    ess_dir.mkdir(parents=True, exist_ok=True)
    
    beta_all = np.vstack(beta_list)
    ess_vals = [effective_sample_size(beta_all[:, j]) for j in range(beta_all.shape[1])]
    
    if hasattr(X, 'columns'):
        param_names = X.columns.tolist()
    else:
        param_names = [f'beta_{j}' for j in range(beta_all.shape[1])]
    
    ess_table = pd.DataFrame({
        'Parameter': param_names,
        'ESS': [round(ess, 1) for ess in ess_vals]
    })
    
    output_file = ess_dir / 'ESS_beta.txt'
    ess_table.to_csv(output_file, sep='\t', index=False)
    
    print(f"\nESS (beta) saved to: {output_file}")
    print(ess_table)
    
    return ess_table


def ess_sigma2_table(sigma2_list, model_name, output_dir='../../r/outputs'):
    # Create output directory
    ess_dir = Path(output_dir) / model_name / 'ESS_tables'
    ess_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine all chains
    sigma2_all = np.concatenate(sigma2_list)
    
    # Compute ESS
    ess_val = round(effective_sample_size(sigma2_all), 1)
    
    # Create table
    ess_table = pd.DataFrame({
        'Parameter': ['Sigma2'],
        'ESS': [ess_val]
    })
    
    # Save ESS table
    output_file = ess_dir / 'ESS_sigma2.txt'
    ess_table.to_csv(output_file, sep='\t', index=False)
    
    print(f"\nESS (sigma^2) saved to: {output_file}")
    print(ess_table)
    
    return ess_table


if __name__ == "__main__":
    print("=" * 60)
    print("CONVERGENCE DETECTION FOR MCMC")
    print("=" * 60)
    
    # Generate synthetic MCMC samples for testing
    np.random.seed(42)
    n_iter = 1000
    n_chains = 3
    p = 3
    
    # Simulate autocorrelated samples (AR(1) process)
    def generate_ar1_samples(n, rho=0.7):
        samples = np.zeros(n)
        samples[0] = np.random.randn()
        for i in range(1, n):
            samples[i] = rho * samples[i-1] + np.random.randn()
        return samples
    
    # Create fake beta samples
    beta_list = []
    sigma2_list = []
    
    for chain in range(n_chains):
        beta_chain = np.column_stack([generate_ar1_samples(n_iter) for _ in range(p)])
        sigma2_chain = np.abs(generate_ar1_samples(n_iter)) + 1.0
        
        beta_list.append(beta_chain)
        sigma2_list.append(sigma2_chain)
    
    print(f"\nTest data generated:")
    print(f"  n_iter = {n_iter}, n_chains = {n_chains}, p = {p}")
    
    # Test ESS calculation
    print("\nTesting ESS calculation...")
    
    # Create fake design matrix
    X_fake = np.random.randn(100, p)
    
    ess_beta_table(beta_list, X_fake, 'test_model')
    ess_sigma2_table(sigma2_list, 'test_model')
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
