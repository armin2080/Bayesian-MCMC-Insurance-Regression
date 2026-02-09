import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import pandas as pd
import time


def gibbs_lm(
    y, X, n_iter=10000, warmup=2000, n_chains=3,
    b0=None, B0=None, a0=0.01, d0=0.01, seed=123
):
    """
    Conjugate Bayesian linear regression with Normal-Inverse-Gamma prior:

      y | beta, sigma2 ~ N(X beta, sigma2 I)
      beta | sigma2    ~ N(b0, sigma2 * (B0)^{-1})
      sigma2           ~ Inv-Gamma(a0, d0)   [shape a0, scale d0]

    Full conditionals:
      beta | sigma2,y  ~ N(bn, sigma2 * (Bn)^{-1})
      sigma2 | beta,y  ~ Inv-Gamma(an, dn)
        dn = d0 + 0.5 * [ (y-Xb)'(y-Xb) + (b-b0)'B0(b-b0) ]
    """

    y = np.asarray(y).reshape(-1)
    X = np.asarray(X)
    n = y.shape[0]
    p = X.shape[1]

    if b0 is None:
        b0 = np.zeros(p)
    if B0 is None:
        B0 = np.eye(p) * 1e-4  # weak prior precision

    b0 = np.asarray(b0).reshape(-1)
    B0 = np.asarray(B0)

    chain_results = []

    # Precompute sufficient statistics
    XtX = X.T @ X
    Xty = X.T @ y

    # Posterior precision is constant across iterations
    Bn = XtX + B0
    bn = np.linalg.solve(Bn, Xty + B0 @ b0)

    # Cholesky factor once (avoid explicit inverse)
    # Bn = L L^T, L lower triangular
    L = np.linalg.cholesky(Bn)

    for chain in range(n_chains):
        rng = np.random.default_rng(seed + chain)

        beta_store = np.zeros((n_iter, p))
        sigma2_store = np.zeros(n_iter)

        beta_draw = np.zeros(p)
        sigma2_draw = 1.0

        t0 = time.time()

        for iter_idx in range(n_iter):
            # ---- beta | sigma2, y ----
            z = rng.standard_normal(p)

            # sample from N(bn, sigma2 * Bn^{-1})
            # if L L^T = Bn, then Bn^{-1} = L^{-T} L^{-1}
            # draw: bn + sqrt(sigma2) * L^{-T} L^{-1} z
            v = np.linalg.solve(L, z)
            w = np.linalg.solve(L.T, v)
            beta_draw = bn + np.sqrt(sigma2_draw) * w

            # ---- sigma2 | beta, y ----
            resid = y - X @ beta_draw
            an = a0 + n / 2.0

            # IMPORTANT: include beta prior quadratic term for sigma2-scaled beta prior
            quad_prior = (beta_draw - b0).T @ B0 @ (beta_draw - b0)
            dn = d0 + 0.5 * (resid @ resid + quad_prior)

            # Inv-Gamma(shape=an, scale=dn): sample via 1/Gamma(an, scale=1/dn)
            sigma2_draw = 1.0 / rng.gamma(shape=an, scale=1.0 / dn)

            beta_store[iter_idx, :] = beta_draw
            sigma2_store[iter_idx] = sigma2_draw

        elapsed = time.time() - t0

        chain_results.append({
            "beta": beta_store[warmup:, :],
            "sigma2": sigma2_store[warmup:],
            "time_elapsed": elapsed
        })

        print(f"Chain {chain + 1} completed. time_elapsed={elapsed:.3f}s")

    return chain_results


def beta_trace_plot(beta_list, model_name, plot_dir='../outputs'):
    n_chains = len(beta_list)
    p = beta_list[0].shape[1]
    output_dir = Path(plot_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_chains))
    
    for j in range(p):
        fig, ax = plt.subplots(figsize=(10, 6))
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
        
        filename = output_dir / f'beta_trace_{j+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")


def sigma2_trace_plot(sigma2_list, model_name, plot_dir='../outputs'):
    n_chains = len(sigma2_list)
    output_dir = Path(plot_dir) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.rainbow(np.linspace(0, 1, n_chains))
    
    fig, ax = plt.subplots(figsize=(10, 6))
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
    
    filename = output_dir / 'Sigma_trace.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


def beta_summary_stats(beta_list):
    beta_all = np.vstack(beta_list)
    summary = {
        'Mean': np.mean(beta_all, axis=0),
        'SD': np.std(beta_all, axis=0, ddof=1),
        '2.5%': np.percentile(beta_all, 2.5, axis=0),
        '97.5%': np.percentile(beta_all, 97.5, axis=0)
    }
    df_summary = pd.DataFrame(summary).round(4)
    print("\nBeta Summary Statistics:")
    print(df_summary)
    return df_summary


def sigma2_summary_stats(sigma2_list):
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
