# ==============================================================================
# Metropolis-Hastings Sampler for Bayesian Linear Regression
# ==============================================================================

library(MASS)
library(coda)

#' Compute log-posterior for linear regression
#' 
#' @param y Response vector
#' @param X Design matrix
#' @param beta Regression coefficients
#' @param sigma2 Error variance
#' @param b0 Prior mean for beta
#' @param B0_inv Prior precision matrix for beta
#' @param a0 Inverse-Gamma shape parameter
#' @param d0 Inverse-Gamma scale parameter
#' @return Log-posterior value
log_posterior <- function(y, X, beta, sigma2, b0, B0_inv, a0, d0) {
  n <- length(y)
  
  # Log-likelihood: y | beta, sigma2 ~ N(X beta, sigma2 I)
  residuals <- y - X %*% beta
  log_lik <- -n/2 * log(2 * pi * sigma2) - sum(residuals^2) / (2 * sigma2)
  
  # Log-prior for beta | sigma2 ~ N(b0, sigma2 * B0^{-1})
  diff_beta <- beta - b0
  log_prior_beta <- -length(beta)/2 * log(2 * pi * sigma2) - 
                    t(diff_beta) %*% B0_inv %*% diff_beta / (2 * sigma2)
  
  # Log-prior for sigma2 ~ IG(a0, d0)
  log_prior_sigma2 <- -a0 * log(d0) - lgamma(a0) - (a0 + 1) * log(sigma2) - d0/sigma2
  
  # Total log-posterior
  log_post <- log_lik + log_prior_beta + log_prior_sigma2
  
  return(as.numeric(log_post))
}


#' Metropolis-Hastings sampler for Bayesian Linear Regression
#'
#' Random Walk Metropolis algorithm with adaptive tuning during warmup.
#' Uses multivariate normal proposal for beta and log-normal for sigma2.
#'
#' @param y Response vector (n x 1)
#' @param X Design matrix (n x p)
#' @param n_iter Total iterations per chain
#' @param warmup Burn-in iterations to discard
#' @param n_chains Number of chains
#' @param b0 Prior mean for beta (default: zeros)
#' @param B0_inv Prior precision matrix for beta (default: 1e4 * I)
#' @param a0 Inverse-Gamma shape (default: 0.01)
#' @param d0 Inverse-Gamma scale (default: 0.01)
#' @param proposal_sd_beta SD for beta proposal (scaled by sqrt(sigma2))
#' @param proposal_sd_sigma2 SD for log(sigma2) proposal
#' @param seed Random seed
#' @return List of chain results
metropolis_hastings_lm <- function(y, X, n_iter = 10000, warmup = 2000, n_chains = 3,
                                   b0 = NULL, B0_inv = NULL, a0 = 0.01, d0 = 0.01,
                                   proposal_sd_beta = 0.1, proposal_sd_sigma2 = 0.1,
                                   seed = 123) {
  
  n <- length(y)
  p <- ncol(X)
  
  # Default priors
  if (is.null(b0)) b0 <- rep(0, p)
  if (is.null(B0_inv)) B0_inv <- diag(1e-4, p)
  
  chain_results <- list()
  
  for (chain in 1:n_chains) {
    set.seed(seed + chain - 1)
    start_time <- Sys.time()
    
    # Storage
    beta_store <- matrix(0, n_iter, p)
    sigma2_store <- numeric(n_iter)
    accept_beta <- 0
    accept_sigma2 <- 0
    
    # Initial values
    beta_current <- rep(0, p)
    sigma2_current <- 1.0
    
    # Compute log-posterior for current state
    log_post_current <- log_posterior(y, X, beta_current, sigma2_current,
                                      b0, B0_inv, a0, d0)
    
    # Adaptive tuning parameters
    tau_beta <- proposal_sd_beta
    tau_sigma2 <- proposal_sd_sigma2
    
    for (iter in 1:n_iter) {
      # ==========================================
      # Step 1: Update beta using Random Walk MH
      # ==========================================
      beta_prop <- beta_current + rnorm(p, 0, tau_beta * sqrt(sigma2_current))
      
      log_post_prop <- log_posterior(y, X, beta_prop, sigma2_current,
                                     b0, B0_inv, a0, d0)
      
      log_alpha <- log_post_prop - log_post_current
      
      if (log(runif(1)) < log_alpha) {
        beta_current <- beta_prop
        log_post_current <- log_post_prop
        accept_beta <- accept_beta + 1
      }
      
      # ==========================================
      # Step 2: Update sigma2 using Random Walk MH  
      # ==========================================
      log_sigma2_prop <- log(sigma2_current) + rnorm(1, 0, tau_sigma2)
      sigma2_prop <- exp(log_sigma2_prop)
      
      log_post_prop <- log_posterior(y, X, beta_current, sigma2_prop,
                                     b0, B0_inv, a0, d0)
      
      # Jacobian correction: |d(exp(x))/dx| = exp(x) = sigma2_prop
      log_alpha <- (log_post_prop - log_post_current) + 
                   (log_sigma2_prop - log(sigma2_current))
      
      if (log(runif(1)) < log_alpha) {
        sigma2_current <- sigma2_prop
        log_post_current <- log_post_prop
        accept_sigma2 <- accept_sigma2 + 1
      }
      
      # Adaptive tuning during warmup
      if (iter <= warmup && iter %% 100 == 0) {
        accept_rate_beta <- accept_beta / iter
        accept_rate_sigma2 <- accept_sigma2 / iter
        
        # Target: ~23.4% for high-dim RWM, ~44% for 1D
        if (accept_rate_beta < 0.20) tau_beta <- tau_beta * 0.9
        if (accept_rate_beta > 0.30) tau_beta <- tau_beta * 1.1
        
        if (accept_rate_sigma2 < 0.35) tau_sigma2 <- tau_sigma2 * 0.9
        if (accept_rate_sigma2 > 0.50) tau_sigma2 <- tau_sigma2 * 1.1
      }
      
      # Store
      beta_store[iter, ] <- beta_current
      sigma2_store[iter] <- sigma2_current
    }
    
    # Compute acceptance rates
    accept_rate_beta <- accept_beta / n_iter
    accept_rate_sigma2 <- accept_sigma2 / n_iter
    time_elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    
    cat(sprintf("Chain %d: β accept=%.1f%%, σ² accept=%.1f%%, time=%.2fs\n",
                chain, accept_rate_beta * 100, accept_rate_sigma2 * 100, time_elapsed))
    
    # Store results (discard warmup)
    chain_results[[chain]] <- list(
      beta = beta_store[(warmup+1):n_iter, ],
      sigma2 = sigma2_store[(warmup+1):n_iter],
      acceptance_rate_beta = accept_rate_beta,
      acceptance_rate_sigma2 = accept_rate_sigma2,
      time_elapsed = time_elapsed
    )
  }
  
  return(chain_results)
}


#' Generate trace plots for Metropolis-Hastings results
#'
#' @param beta_list List of beta matrices from chains
#' @param feature_names Names of features
#' @param model_name Model name for file naming
#' @param plot_dir Output directory
beta_trace_plot_mh <- function(beta_list, feature_names = NULL, 
                               model_name = "mh_baseline", plot_dir = "../outputs") {
  
  n_chains <- length(beta_list)
  n_params <- ncol(beta_list[[1]])
  
  if (is.null(feature_names)) {
    feature_names <- paste0("beta_", 0:(n_params-1))
  }
  
  output_path <- file.path(plot_dir, model_name)
  dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
  
  colors <- c("blue", "red", "green", "purple")
  
  for (j in 1:n_params) {
    png(file.path(output_path, sprintf("beta_trace_%d.png", j)), 
        width = 800, height = 600, res = 100)
    
    par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
    
    # Trace plot
    plot(NULL, xlim = c(1, nrow(beta_list[[1]])), 
         ylim = range(sapply(beta_list, function(x) range(x[, j]))),
         xlab = "Iteration", ylab = feature_names[j],
         main = sprintf("Trace: %s", feature_names[j]))
    
    for (chain in 1:n_chains) {
      lines(beta_list[[chain]][, j], col = colors[chain], lwd = 0.5)
    }
    legend("topright", legend = paste("Chain", 1:n_chains), 
           col = colors[1:n_chains], lwd = 2)
    
    # Density plot
    all_samples <- do.call(c, lapply(beta_list, function(x) x[, j]))
    plot(density(all_samples), main = sprintf("Density: %s", feature_names[j]),
         xlab = feature_names[j], col = "darkblue", lwd = 2)
    
    dev.off()
  }
  
  cat(sprintf("✓ MH trace plots saved to %s\n", output_path))
}


#' Generate sigma2 trace plot
#'
#' @param sigma2_list List of sigma2 vectors from chains
#' @param model_name Model name
#' @param plot_dir Output directory
sigma2_trace_plot_mh <- function(sigma2_list, model_name = "mh_baseline", 
                                 plot_dir = "../outputs") {
  
  n_chains <- length(sigma2_list)
  output_path <- file.path(plot_dir, model_name)  
  dir.create(output_path, recursive = TRUE, showWarnings = FALSE)
  
  colors <- c("blue", "red", "green", "purple")
  
  png(file.path(output_path, "Sigma_trace.png"), 
      width = 800, height = 600, res = 100)
  
  par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))
  
  # Trace plot
  plot(NULL, xlim = c(1, length(sigma2_list[[1]])),
       ylim = range(sapply(sigma2_list, range)),
       xlab = "Iteration", ylab = expression(sigma^2),
       main = expression(paste("Trace: ", sigma^2)))
  
  for (chain in 1:n_chains) {
    lines(sigma2_list[[chain]], col = colors[chain], lwd = 0.5)
  }
  legend("topright", legend = paste("Chain", 1:n_chains),
         col = colors[1:n_chains], lwd = 2)
  
  # Density plot
  all_samples <- do.call(c, sigma2_list)
  plot(density(all_samples), main = expression(paste("Density: ", sigma^2)),
       xlab = expression(sigma^2), col = "darkblue", lwd = 2)
  
  dev.off()
  
  cat(sprintf("✓ MH σ² trace plot saved to %s\n", output_path))
}


# ==============================================================================
# Example Usage / Testing
# ==============================================================================

if (FALSE) {
  # Generate test data
  set.seed(42)
  n <- 100
  p <- 3
  X <- cbind(1, matrix(rnorm(n * (p-1)), n, p-1))
  beta_true <- c(2, 1.5, -1)
  sigma2_true <- 1
  y <- X %*% beta_true + rnorm(n, 0, sqrt(sigma2_true))
  
  # Run MH sampler
  cat("Running Metropolis-Hastings sampler...\n")
  mh_results <- metropolis_hastings_lm(
    y = y,
    X = X,
    n_iter = 10000,
    warmup = 2000,
    n_chains = 3,
    seed = 123
  )
  
  # Generate trace plots
  beta_list <- lapply(mh_results, function(x) x$beta)
  sigma2_list <- lapply(mh_results, function(x) x$sigma2)
  
  beta_trace_plot_mh(beta_list, feature_names = c("Intercept", "X1", "X2"))
  sigma2_trace_plot_mh(sigma2_list)
  
  # Print summaries
  cat("\n", "="*60, "\n", sep="")
  cat("SUMMARY STATISTICS\n")
  cat("="*60, "\n", sep="")
  
  beta_combined <- do.call(rbind, beta_list)
  cat("\nBeta posteriors:\n")
  print(summary(beta_combined))
  
  sigma2_combined <- do.call(c, sigma2_list)
  cat("\nSigma^2 posterior:\n")
  print(summary(sigma2_combined))
}
