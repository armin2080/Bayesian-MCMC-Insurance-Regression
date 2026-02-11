# ==============================================================================
# Algorithm Comparison: Gibbs Sampling vs Metropolis-Hastings
# ==============================================================================

library(coda)
library(ggplot2)
library(gridExtra)

#' Compute Effective Sample Size (ESS)
#'
#' @param samples Vector or matrix of MCMC samples
#' @return ESS value
compute_ess <- function(samples) {
  if (is.matrix(samples)) {
    # Multiple chains: convert to mcmc.list
    mcmc_list <- mcmc.list(lapply(1:nrow(samples), function(i) mcmc(samples[i,])))
    ess <- effectiveSize(mcmc_list)
  } else {
    # Single chain
    mcmc_obj <- mcmc(samples)
    ess <- effectiveSize(mcmc_obj)
  }
  return(as.numeric(ess))
}


#' Compute Gelman-Rubin R-hat statistic
#'
#' @param chains_list List of chains for a parameter
#' @return R-hat value
compute_rhat <- function(chains_list) {
  mcmc_list <- mcmc.list(lapply(chains_list, mcmc))
  gr <- gelman.diag(mcmc_list, autoburnin = FALSE)
  return(gr$psrf[1, 1])
}


#' Compare Gibbs vs Metropol is-Hastings algorithms
#'
#' @param gibbs_results List of results from Gibbs sampler
#' @param mh_results List of results from MH sampler
#' @param param_names Character vector of parameter names
#' @param output_dir Output directory for results
compare_algorithms <- function(gibbs_results, mh_results,
                               model_name = "baseline",
                              param_names = NULL, 
                              output_dir = "../outputs/algorithm_comparison") {
  
  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Extract samples
  gibbs_beta <- lapply(gibbs_results, function(x) x$beta)
  mh_beta <- lapply(mh_results, function(x) x$beta)
  
  gibbs_sigma2 <- lapply(gibbs_results, function(x) x$sigma2)
  mh_sigma2 <- lapply(mh_results, function(x) x$sigma2)
  
  n_chains <- length(gibbs_beta)
  n_params <- ncol(gibbs_beta[[1]])
  
  if (is.null(param_names)) {
    param_names <- paste0("β", 0:(n_params-1))
  }
  
  # Get timing
  gibbs_time <- sum(sapply(gibbs_results, function(x) x$time_elapsed))
  mh_time <- sum(sapply(mh_results, function(x) x$time_elapsed))
  
  cat("==============================================================================\n")
  cat("ALGORITHM COMPARISON: Gibbs Sampling vs Metropolis-Hastings\n")
  cat("==============================================================================\n\n")
  
  # ==========================================
  # 1. ESS Comparison
  # ==========================================
  cat("1. EFFECTIVE SAMPLE SIZE (ESS) COMPARISON\n")
  cat("------------------------------------------------------------------------------\n")
  
  ess_comparison <- data.frame()
  
  for (j in 1:n_params) {
    # Gibbs ESS
    gibbs_param <- do.call(rbind, lapply(gibbs_beta, function(x) x[, j]))
    gibbs_ess <- compute_ess(gibbs_param)
    
    # MH ESS
    mh_param <- do.call(rbind, lapply(mh_beta, function(x) x[, j]))
    mh_ess <- compute_ess(mh_param)
    
    ess_comparison <- rbind(ess_comparison, data.frame(
      Parameter = param_names[j],
      Gibbs_ESS = gibbs_ess,
      MH_ESS = mh_ess,
      ESS_Ratio = gibbs_ess / mh_ess
    ))
  }
  
  # Add sigma2
  gibbs_sigma2_arr <- do.call(rbind, lapply(gibbs_sigma2, function(x) matrix(x, 1)))
  mh_sigma2_arr <- do.call(rbind, lapply(mh_sigma2, function(x) matrix(x, 1)))
  
  gibbs_sigma2_ess <- compute_ess(gibbs_sigma2_arr)
  mh_sigma2_ess <- compute_ess(mh_sigma2_arr)
  
  ess_comparison <- rbind(ess_comparison, data.frame(
    Parameter = "σ²",
    Gibbs_ESS = gibbs_sigma2_ess,
    MH_ESS = mh_sigma2_ess,
    ESS_Ratio = gibbs_sigma2_ess / mh_sigma2_ess
  ))
  
  print(ess_comparison)
  write.csv(ess_comparison, file.path(output_dir, "ess_comparison.csv"), row.names = FALSE)
  
  cat(sprintf("\nAverage ESS Ratio: %.1fx\n", mean(ess_comparison$ESS_Ratio)))
  
  # ==========================================
  # 2. Computational Efficiency
  # ==========================================
  cat("\n2. COMPUTATIONAL EFFICIENCY (ESS per second)\n")
  cat("------------------------------------------------------------------------------\n")
  
  gibbs_avg_ess <- mean(ess_comparison$Gibbs_ESS)
  mh_avg_ess <- mean(ess_comparison$MH_ESS)
  
  gibbs_ess_per_sec <- gibbs_avg_ess / gibbs_time
  mh_ess_per_sec <- mh_avg_ess / mh_time
  
  efficiency_ratio <- gibbs_ess_per_sec / mh_ess_per_sec
  
  efficiency_df <- data.frame(
    Algorithm = c("Gibbs", "Metropolis-Hastings"),
    Total_Time_sec = c(gibbs_time, mh_time),
    Avg_ESS = c(gibbs_avg_ess, mh_avg_ess),
    ESS_per_sec = c(gibbs_ess_per_sec, mh_ess_per_sec)
  )
  
  print(efficiency_df)
  write.csv(efficiency_df, file.path(output_dir, "efficiency_comparison.csv"), row.names = FALSE)
  
  cat(sprintf("\nEfficiency Ratio (Gibbs/MH): %.1fx\n", efficiency_ratio))
  
  # ==========================================
  # 3. Convergence Diagnostics (R-hat)
  # ==========================================
  cat("\n3. CONVERGENCE DIAGNOSTICS (R-hat)\n")
  cat("------------------------------------------------------------------------------\n")
  
  rhat_comparison <- data.frame()
  
  for (j in 1:n_params) {
    gibbs_chains <- lapply(gibbs_beta, function(x) x[, j])
    mh_chains <- lapply(mh_beta, function(x) x[, j])
    
    gibbs_rhat <- compute_rhat(gibbs_chains)
    mh_rhat <- compute_rhat(mh_chains)
    
    rhat_comparison <- rbind(rhat_comparison, data.frame(
      Parameter = param_names[j],
      Gibbs_Rhat = gibbs_rhat,
      MH_Rhat = mh_rhat,
      Both_Converged = (gibbs_rhat < 1.1) && (mh_rhat < 1.1)
    ))
  }
  
  # Add sigma2
  gibbs_rhat_sigma2 <- compute_rhat(gibbs_sigma2)
  mh_rhat_sigma2 <- compute_rhat(mh_sigma2)
  
  rhat_comparison <- rbind(rhat_comparison, data.frame(
    Parameter = "σ²",
    Gibbs_Rhat = gibbs_rhat_sigma2,
    MH_Rhat = mh_rhat_sigma2,
    Both_Converged = (gibbs_rhat_sigma2 < 1.1) && (mh_rhat_sigma2 < 1.1)
  ))
  
  print(rhat_comparison)
  write.csv(rhat_comparison, file.path(output_dir, "rhat_comparison.csv"), row.names = FALSE)
  
  # ==========================================
  # 4. Posterior Agreement
  # ==========================================
  cat("\n4. POSTERIOR ESTIMATES AGREEMENT\n")
  cat("------------------------------------------------------------------------------\n")
  
  posterior_comparison <- data.frame()
  
  for (j in 1:n_params) {
    gibbs_param <- do.call(c, lapply(gibbs_beta, function(x) x[, j]))
    mh_param <- do.call(c, lapply(mh_beta, function(x) x[, j]))
    
    posterior_comparison <- rbind(posterior_comparison, data.frame(
      Parameter = param_names[j],
      Gibbs_Mean = mean(gibbs_param),
      MH_Mean = mean(mh_param),
      Difference = mean(gibbs_param) - mean(mh_param),
      Gibbs_SD = sd(gibbs_param),
      MH_SD = sd(mh_param)
    ))
  }
  
  # Add sigma2
  gibbs_sigma2_all <- do.call(c, gibbs_sigma2)
  mh_sigma2_all <- do.call(c, mh_sigma2)
  
  posterior_comparison <- rbind(posterior_comparison, data.frame(
    Parameter = "σ²",
    Gibbs_Mean = mean(gibbs_sigma2_all),
    MH_Mean = mean(mh_sigma2_all),
    Difference = mean(gibbs_sigma2_all) - mean(mh_sigma2_all),
    Gibbs_SD = sd(gibbs_sigma2_all),
    MH_SD = sd(mh_sigma2_all)
  ))
  
  print(posterior_comparison)
  write.csv(posterior_comparison, file.path(output_dir, "posterior_estimates_comparison.csv"), 
            row.names = FALSE)
  
  cat(sprintf("\nMax absolute difference: %.6f\n", max(abs(posterior_comparison$Difference))))
  
  # ==========================================
  # 5. Acceptance Rates (MH only)
  # ==========================================
  cat("\n5. METROPOLIS-HASTINGS ACCEPTANCE RATES\n")
  cat("------------------------------------------------------------------------------\n")
  
  mh_accept_beta <- mean(sapply(mh_results, function(x) x$acceptance_rate_beta))
  mh_accept_sigma2 <- mean(sapply(mh_results, function(x) x$acceptance_rate_sigma2))
  
  acceptance_df <- data.frame(
    Component = c("Beta", "Sigma²"),
    Acceptance_Rate = c(mh_accept_beta, mh_accept_sigma2),
    Target_Rate = c(0.234, 0.44),
    On_Target = c(
      abs(mh_accept_beta - 0.234) < 0.05,
      abs(mh_accept_sigma2 - 0.44) < 0.10
    )
  )
  
  print(acceptance_df)
  write.csv(acceptance_df, file.path(output_dir, "mh_acceptance_rates.csv"), row.names = FALSE)
  
  # ==========================================
  # Summary Plot
  # ==========================================
  png(file.path(output_dir, "algorithm_comparison_summary.png"),
      width = 1200, height = 400, res = 100)
  
  par(mfrow = c(1, 3), mar = c(5, 4, 4, 2))
  
  # Plot 1: ESS comparison
  barplot(rbind(ess_comparison$Gibbs_ESS, ess_comparison$MH_ESS),
          beside = TRUE,
          names.arg = ess_comparison$Parameter,
          col = c("steelblue", "coral"),
          main = "Effective Sample Size",
          ylab = "ESS",
          legend.text = c("Gibbs", "MH"),
          las = 2,
          cex.names = 0.8)
  
  # Plot 2: Efficiency comparison
  barplot(c(gibbs_ess_per_sec, mh_ess_per_sec),
          names.arg = c("Gibbs", "MH"),
          col = c("steelblue", "coral"),
          main = "Computational Efficiency",
          ylab = "ESS per second")
  
  # Plot 3: R-hat comparison
  barplot(rbind(rhat_comparison$Gibbs_Rhat, rhat_comparison$MH_Rhat),
          beside = TRUE,
          names.arg = rhat_comparison$Parameter,
          col = c("steelblue", "coral"),
          main = "Convergence (R-hat)",
          ylab = "R-hat",
          legend.text = c("Gibbs", "MH"),
          las = 2,
          cex.names = 0.8)
  abline(h = 1.1, lty = 2, col = "red", lwd = 2)
  
  dev.off()
  
  cat(sprintf("\n✓ Comparison plots saved to %s\n", output_dir))
  
  # ==========================================
  # Final Summary
  # ==========================================
  cat("\n==============================================================================\n")
  cat("SUMMARY\n")
  cat("==============================================================================\n")
  cat(sprintf("Gibbs Sampling:\n"))
  cat(sprintf("  - Average ESS: %.0f\n", gibbs_avg_ess))
  cat(sprintf("  - ESS/second: %.1f\n", gibbs_ess_per_sec))
  cat(sprintf("  - All converged: %s\n", ifelse(all(rhat_comparison$Gibbs_Rhat < 1.1), "✓", "✗")))
  
  cat(sprintf("\nMetropolis-Hastings:\n"))
  cat(sprintf("  - Average ESS: %.0f\n", mh_avg_ess))
  cat(sprintf("  - ESS/second: %.1f\n", mh_ess_per_sec))
  cat(sprintf("  - All converged: %s\n", ifelse(all(rhat_comparison$MH_Rhat < 1.1), "✓", "✗")))
  cat(sprintf("  - β acceptance: %.1f%%\n", mh_accept_beta * 100))
  cat(sprintf("  - σ² acceptance: %.1f%%\n", mh_accept_sigma2 * 100))
  
  cat(sprintf("\nEfficiency Advantage: Gibbs is %.1fx faster (ESS/second)\n", efficiency_ratio))
  cat("==============================================================================\n")
}


# ==============================================================================
# Example Usage
# ==============================================================================

if (FALSE) {
  # Load Gibbs and MH results
  source("Gibbs_Sampling.R")
  source("Metropolis_Hastings.R")
  
  # Generate test data
  set.seed(42)
  n <- 100
  p <- 3
  X <- cbind(1, matrix(rnorm(n * (p-1)), n, p-1))
  beta_true <- c(2, 1.5, -1)
  sigma2_true <- 1
  y <- X %*% beta_true + rnorm(n, 0, sqrt(sigma2_true))
  
  # Run both samplers
  gibbs_results <- gibbs_lm(y, X, n_iter = 10000, warmup = 2000, n_chains = 3)
  mh_results <- metropolis_hastings_lm(y, X, n_iter = 10000, warmup = 2000, n_chains = 3)
  
  # Compare algorithms
  compare_algorithms(
    gibbs_results = gibbs_results,
    mh_results = mh_results,
    param_names = c("Intercept", "X1", "X2"),
    output_dir = "../outputs/algorithm_comparison"
  )
}
