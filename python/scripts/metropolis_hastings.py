import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import pandas as pd
import time


def metropolis_hastings_lm(y, X, n_iter=10000, warmup=2000, n_chains=4,
                           b0=None, B0=None, a0=0.01, d0=0.01, 
                           proposal_sd_beta=0.1, proposal_sd_sigma2=0.1,
                           seed=123):
    """
    Random walk Metropolis-Hastings for Bayesian linear regression.
    Beta and sigma2 are updated one at a time with Gaussian proposals.
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
        
        beta_store = np.zeros((n_iter, p))
        sigma2_store = np.zeros(n_iter)
        accept_beta = 0
        accept_sigma2 = 0
        
        beta_current = np.zeros(p)
        sigma2_current = 1.0
        
        log_post_current = log_posterior(y, X, beta_current, sigma2_current, 
                                         b0, B0, a0, d0)
        
        for iter_idx in range(n_iter):
            # propose new beta
            beta_prop = beta_current + np.random.normal(0, proposal_sd_beta * np.sqrt(sigma2_current), p)
            
            log_post_prop = log_posterior(y, X, beta_prop, sigma2_current, 
                                         b0, B0, a0, d0)
            
            log_alpha = log_post_prop - log_post_current
            
            if np.log(np.random.uniform()) < log_alpha:
                beta_current = beta_prop
                log_post_current = log_post_prop
                accept_beta += 1
            
            # propose new sigma2 (on log scale to keep it positive)
            log_sigma2_current = np.log(sigma2_current)
            log_sigma2_prop = log_sigma2_current + np.random.normal(0, proposal_sd_sigma2)
            sigma2_prop = np.exp(log_sigma2_prop)
            
            log_post_prop = log_posterior(y, X, beta_current, sigma2_prop, 
                                         b0, B0, a0, d0)
            
            # account for Jacobian when working on log scale
            log_alpha = (log_post_prop - log_post_current) + (log_sigma2_prop - log_sigma2_current)
            
            if np.log(np.random.uniform()) < log_alpha:
                sigma2_current = sigma2_prop
                log_post_current = log_post_prop
                accept_sigma2 += 1
            
            beta_store[iter_idx, :] = beta_current
            sigma2_store[iter_idx] = sigma2_current
        
        elapsed_time = time.time() - start_time
        
        accept_rate_beta = accept_beta / n_iter
        accept_rate_sigma2 = accept_sigma2 / n_iter
        
        chain_results.append({
            'beta': beta_store[warmup:, :],
            'sigma2': sigma2_store[warmup:],
            'acceptance_rate_beta': accept_rate_beta,
            'acceptance_rate_sigma2': accept_rate_sigma2,
            'time_elapsed': elapsed_time
        })
        
        print(f"Chain {chain + 1}: beta_accept={accept_rate_beta:.3f}, sigma2_accept={accept_rate_sigma2:.3f}, time={elapsed_time:.2f}s")
    
    return chain_results


def log_posterior(y, X, beta, sigma2, b0, B0, a0, d0):
    """Log posterior up to a constant."""
    n = len(y)
    p = len(beta)
    
    resid = y - X @ beta
    log_lik = -n/2 * np.log(2 * np.pi * sigma2) - 1/(2*sigma2) * np.sum(resid**2)
    
    beta_centered = beta - b0
    log_prior_beta = -p/2 * np.log(2 * np.pi * sigma2) - 1/(2*sigma2) * (beta_centered @ B0 @ beta_centered)
    
    log_prior_sigma2 = -(a0 + 1) * np.log(sigma2) - d0 / sigma2
    
    return log_lik + log_prior_beta + log_prior_sigma2


def adaptive_mh_lm(y, X, n_iter=10000, warmup=2000, n_chains=4,
                   b0=None, B0=None, a0=0.01, d0=0.01,
                   initial_proposal_sd_beta=0.1, initial_proposal_sd_sigma2=0.1,
                   target_accept=0.234, adapt_interval=50, seed=123):
    """
    Adaptive MH that tunes proposal scales during warmup to hit a target acceptance rate.
    Typically aim for ~23% acceptance for high-dimensional random walk samplers.
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
        
        beta_store = np.zeros((n_iter, p))
        sigma2_store = np.zeros(n_iter)
        accept_beta = 0
        accept_sigma2 = 0
        
        proposal_sd_beta = initial_proposal_sd_beta
        proposal_sd_sigma2 = initial_proposal_sd_sigma2
        
        beta_current = np.zeros(p)
        sigma2_current = 1.0
        log_post_current = log_posterior(y, X, beta_current, sigma2_current, 
                                         b0, B0, a0, d0)
        
        window_accept_beta = 0
        window_accept_sigma2 = 0
        
        for iter_idx in range(n_iter):
            # update beta
            beta_prop = beta_current + np.random.normal(0, proposal_sd_beta * np.sqrt(sigma2_current), p)
            log_post_prop = log_posterior(y, X, beta_prop, sigma2_current, b0, B0, a0, d0)
            log_alpha = log_post_prop - log_post_current
            
            if np.log(np.random.uniform()) < log_alpha:
                beta_current = beta_prop
                log_post_current = log_post_prop
                accept_beta += 1
                window_accept_beta += 1
            
            # update sigma2
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
            
            beta_store[iter_idx, :] = beta_current
            sigma2_store[iter_idx] = sigma2_current
            
            # adjust proposal scales during warmup
            if iter_idx < warmup and (iter_idx + 1) % adapt_interval == 0:
                accept_rate_beta = window_accept_beta / adapt_interval
                accept_rate_sigma2 = window_accept_sigma2 / adapt_interval
                
                if accept_rate_beta > target_accept:
                    proposal_sd_beta *= 1.1
                else:
                    proposal_sd_beta *= 0.9
                
                if accept_rate_sigma2 > target_accept:
                    proposal_sd_sigma2 *= 1.1
                else:
                    proposal_sd_sigma2 *= 0.9
                
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
        
        print(f"Chain {chain + 1} (adaptive): beta_accept={accept_beta / n_iter:.3f}, "
              f"sigma2_accept={accept_sigma2 / n_iter:.3f}, time={elapsed_time:.2f}s")
    
    return chain_results


def mh_trace_plot(beta_list, model_name, plot_dir='../outputs'):
    """Trace plots for MH samples."""
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
