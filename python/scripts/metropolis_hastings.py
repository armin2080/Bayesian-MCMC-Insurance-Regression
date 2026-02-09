import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import pandas as pd
import time


def metropolis_hastings_lm(y, X, n_iter=10000, warmup=2000, n_chains=3,
                           b0=None, B0=None, a0=0.01, d0=0.01, 
                           proposal_sd_beta=0.1, proposal_sd_sigma2=0.1,
                           seed=123):
    """
    Metropolis-Hastings sampler for Bayesian Linear Regression.
    
    This is a Random Walk Metropolis algorithm where we sample beta and sigma2 jointly.
    We use a multivariate normal proposal for beta and log-normal proposal for sigma2.
    
    Parameters:
    -----------
    y : array (n,)
        Response vector
    X : array (n, p)
        Design matrix
    n_iter : int
        Total iterations per chain
    warmup : int
        Burn-in iterations to discard
    n_chains : int
        Number of chains to run
    b0 : array (p,)
        Prior mean for beta (default: zeros)
    B0 : array (p, p)
        Prior precision matrix for beta (default: 1e-4 * I)
    a0, d0 : float
        Inverse-Gamma prior parameters for sigma2
    proposal_sd_beta : float
        Standard deviation for beta proposal (will be scaled by sqrt(sigma2))
    proposal_sd_sigma2 : float
        Standard deviation for log(sigma2) proposal
    seed : int
        Random seed
        
    Returns:
    --------
    chain_results : list of dicts
        Each dict contains 'beta', 'sigma2', 'acceptance_rate', and 'time_elapsed'
    """
    
    n = len(y)
    p = X.shape[1]
    
    # Default priors (weakly informative)
    if b0 is None:
        b0 = np.zeros(p)
    if B0 is None:
        B0 = np.eye(p) * 1e-4
    
    chain_results = []
    
    for chain in range(n_chains):
        start_time = time.time()
        np.random.seed(seed + chain)
        
        # Storage
        beta_store = np.zeros((n_iter, p))
        sigma2_store = np.zeros(n_iter)
        accept_beta = 0
        accept_sigma2 = 0
        
        # Initial values
        beta_current = np.zeros(p)  # Start at prior mean
        sigma2_current = 1.0
        
        # Compute log-posterior for current state
        log_post_current = log_posterior(y, X, beta_current, sigma2_current, 
                                         b0, B0, a0, d0)
        
        for iter_idx in range(n_iter):
            # ==========================================
            # Step 1: Update beta using Random Walk MH
            # ==========================================
            # Proposal: beta_prop ~ N(beta_current, proposal_sd_beta^2 * sigma2_current * I)
            beta_prop = beta_current + np.random.normal(0, proposal_sd_beta * np.sqrt(sigma2_current), p)
            
            # Compute log-posterior for proposed beta
            log_post_prop = log_posterior(y, X, beta_prop, sigma2_current, 
                                         b0, B0, a0, d0)
            
            # Acceptance ratio (log scale)
            log_alpha = log_post_prop - log_post_current
            
            # Accept/reject
            if np.log(np.random.uniform()) < log_alpha:
                beta_current = beta_prop
                log_post_current = log_post_prop
                accept_beta += 1
            
            # =============================================
            # Step 2: Update sigma2 using Random Walk MH
            # =============================================
            # Proposal on log scale: log(sigma2_prop) ~ N(log(sigma2_current), proposal_sd_sigma2^2)
            log_sigma2_current = np.log(sigma2_current)
            log_sigma2_prop = log_sigma2_current + np.random.normal(0, proposal_sd_sigma2)
            sigma2_prop = np.exp(log_sigma2_prop)
            
            # Compute log-posterior for proposed sigma2
            log_post_prop = log_posterior(y, X, beta_current, sigma2_prop, 
                                         b0, B0, a0, d0)
            
            # Acceptance ratio (including Jacobian for log transformation)
            # Jacobian: d(sigma2)/d(log_sigma2) = sigma2
            log_alpha = (log_post_prop - log_post_current) + (log_sigma2_prop - log_sigma2_current)
            
            # Accept/reject
            if np.log(np.random.uniform()) < log_alpha:
                sigma2_current = sigma2_prop
                log_post_current = log_post_prop
                accept_sigma2 += 1
            
            # Store current state
            beta_store[iter_idx, :] = beta_current
            sigma2_store[iter_idx] = sigma2_current
        
        elapsed_time = time.time() - start_time
        
        # Acceptance rates
        accept_rate_beta = accept_beta / n_iter
        accept_rate_sigma2 = accept_sigma2 / n_iter
        
        chain_results.append({
            'beta': beta_store[warmup:, :],
            'sigma2': sigma2_store[warmup:],
            'acceptance_rate_beta': accept_rate_beta,
            'acceptance_rate_sigma2': accept_rate_sigma2,
            'time_elapsed': elapsed_time
        })
        
        print(f"Chain {chain + 1} completed:")
        print(f"  - Beta acceptance rate: {accept_rate_beta:.3f}")
        print(f"  - Sigma2 acceptance rate: {accept_rate_sigma2:.3f}")
        print(f"  - Time elapsed: {elapsed_time:.2f}s")
    
    return chain_results


def log_posterior(y, X, beta, sigma2, b0, B0, a0, d0):
    """
    Compute log posterior: log p(beta, sigma2 | y) up to a constant.
    
    log p(beta, sigma2 | y) = log p(y | beta, sigma2) + log p(beta | sigma2) + log p(sigma2)
    
    Where:
    - log p(y | beta, sigma2) = -n/2 * log(2*pi*sigma2) - 1/(2*sigma2) * ||y - X*beta||^2
    - log p(beta | sigma2) = -p/2 * log(2*pi*sigma2) - 1/(2*sigma2) * (beta - b0)' B0 (beta - b0)
    - log p(sigma2) = IG(a0, d0) = -(a0 + 1)*log(sigma2) - d0/sigma2 + const
    """
    n = len(y)
    p = len(beta)
    
    # Likelihood: log p(y | beta, sigma2)
    resid = y - X @ beta
    log_lik = -n/2 * np.log(2 * np.pi * sigma2) - 1/(2*sigma2) * np.sum(resid**2)
    
    # Prior for beta: log p(beta | sigma2)
    beta_centered = beta - b0
    log_prior_beta = -p/2 * np.log(2 * np.pi * sigma2) - 1/(2*sigma2) * (beta_centered @ B0 @ beta_centered)
    
    # Prior for sigma2: log p(sigma2) ~ IG(a0, d0)
    log_prior_sigma2 = -(a0 + 1) * np.log(sigma2) - d0 / sigma2
    
    return log_lik + log_prior_beta + log_prior_sigma2


def adaptive_mh_lm(y, X, n_iter=10000, warmup=2000, n_chains=3,
                   b0=None, B0=None, a0=0.01, d0=0.01,
                   initial_proposal_sd_beta=0.1, initial_proposal_sd_sigma2=0.1,
                   target_accept=0.234, adapt_interval=50, seed=123):
    """
    Adaptive Metropolis-Hastings with automatic tuning of proposal distributions.
    
    During warmup, proposal standard deviations are adjusted to achieve target acceptance rate.
    This is a simplified adaptive scheme that adjusts every 'adapt_interval' iterations.
    
    Target acceptance rate of 0.234 is optimal for high-dimensional random walk MH.
    """
    
    n = len(y)
    p = X.shape[1]
    
    if b0 is None:
        b0 = np.zeros(p)
    if B0 is None:
        B0 = np.eye(p) * 1e-4
    
    chain_results = []
    
    for chain in range(n_chains):
        start_time = time.time()
        np.random.seed(seed + chain)
        
        # Storage
        beta_store = np.zeros((n_iter, p))
        sigma2_store = np.zeros(n_iter)
        accept_beta = 0
        accept_sigma2 = 0
        
        # Adaptive proposal parameters
        proposal_sd_beta = initial_proposal_sd_beta
        proposal_sd_sigma2 = initial_proposal_sd_sigma2
        
        # Initial values
        beta_current = np.zeros(p)
        sigma2_current = 1.0
        log_post_current = log_posterior(y, X, beta_current, sigma2_current, 
                                         b0, B0, a0, d0)
        
        # Track acceptance in windows for adaptation
        window_accept_beta = 0
        window_accept_sigma2 = 0
        
        for iter_idx in range(n_iter):
            # Update beta
            beta_prop = beta_current + np.random.normal(0, proposal_sd_beta * np.sqrt(sigma2_current), p)
            log_post_prop = log_posterior(y, X, beta_prop, sigma2_current, b0, B0, a0, d0)
            log_alpha = log_post_prop - log_post_current
            
            if np.log(np.random.uniform()) < log_alpha:
                beta_current = beta_prop
                log_post_current = log_post_prop
                accept_beta += 1
                window_accept_beta += 1
            
            # Update sigma2
            log_sigma2_current = np.log(sigma2_current)
            log_sigma2_prop = log_sigma2_current + np.random.normal(0, proposal_sd_sigma2)
            sigma2_prop = np.exp(log_sigma2_prop)
            log_post_prop = log_posterior(y, X, beta_current, sigma2_prop, b0, B0, a0, d0)
            log_alpha = (log_post_prop - log_post_current) + (log_sigma2_prop - log_sigma2_current)
            
            if np.log(np.random.uniform()) < log_alpha:
                sigma2_current = sigma2_prop
                log_post_current = log_post_prop
                accept_sigma2 += 1
                window_accept_sigma2 += 1
            
            # Store
            beta_store[iter_idx, :] = beta_current
            sigma2_store[iter_idx] = sigma2_current
            
            # Adapt proposal during warmup
            if iter_idx < warmup and (iter_idx + 1) % adapt_interval == 0:
                accept_rate_beta = window_accept_beta / adapt_interval
                accept_rate_sigma2 = window_accept_sigma2 / adapt_interval
                
                # Adjust proposal SDs (Roberts & Rosenthal adaptive scheme)
                if accept_rate_beta > target_accept:
                    proposal_sd_beta *= 1.1  # Increase step size
                else:
                    proposal_sd_beta *= 0.9  # Decrease step size
                
                if accept_rate_sigma2 > target_accept:
                    proposal_sd_sigma2 *= 1.1
                else:
                    proposal_sd_sigma2 *= 0.9
                
                # Reset window counters
                window_accept_beta = 0
                window_accept_sigma2 = 0
        
        elapsed_time = time.time() - start_time
        
        chain_results.append({
            'beta': beta_store[warmup:, :],
            'sigma2': sigma2_store[warmup:],
            'acceptance_rate_beta': accept_beta / n_iter,
            'acceptance_rate_sigma2': accept_sigma2 / n_iter,
            'final_proposal_sd_beta': proposal_sd_beta,
            'final_proposal_sd_sigma2': proposal_sd_sigma2,
            'time_elapsed': elapsed_time
        })
        
        print(f"Chain {chain + 1} completed (Adaptive MH):")
        print(f"  - Beta acceptance rate: {accept_beta / n_iter:.3f}")
        print(f"  - Sigma2 acceptance rate: {accept_sigma2 / n_iter:.3f}")
        print(f"  - Final proposal SD beta: {proposal_sd_beta:.4f}")
        print(f"  - Final proposal SD sigma2: {proposal_sd_sigma2:.4f}")
        print(f"  - Time elapsed: {elapsed_time:.2f}s")
    
    return chain_results


def mh_trace_plot(beta_list, model_name, plot_dir='../outputs'):
    """Create trace plots for MH samples (same format as Gibbs)."""
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
        ax.set_title(rf'MH Trace plot: $\beta_{{{j}}}$', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        filename = output_dir / f'beta_trace_{j+1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
