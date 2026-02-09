# ==============================================================================
# Master Script: Run Complete Bayesian MCMC Analysis
# ==============================================================================
# This script orchestrates the complete analysis pipeline:
# 1. Data preprocessing
# 2. Baseline Gibbs sampling analysis
# 3. Metropolis-Hastings analysis
# 4. Algorithm comparison
# 5. Residual diagnostics
# 6. Comprehensive reporting
# ==============================================================================

# Set up paths
setwd(dirname(sys.frame(1)$ofile))  # Set working directory to script location

# Load required libraries
suppressPackageStartupMessages({
  library(MASS)
  library(coda)
  library(ggplot2)
  library(gridExtra)
})

# Source all analysis scripts
source("Data_Preprocessing.R")
source("Model_Setup.R")
source("Gibbs_Sampling.R")
source("Metropolis_Hastings.R")
source("Convergence_Detection.R")
source("Posterior_Inference.R")
source("Algorithm_Comparison.R")
source("Residual_Diagnostics.R")

cat("\n")
cat("╔"*80, "\n", sep="")
cat("║ BAYESIAN MCMC ANALYSIS: MEDICAL INSURANCE COSTS\n")
cat("║ Complete Analysis Pipeline\n")
cat("╚"*80, "\n\n", sep="")

# Start timing
analysis_start_time <- Sys.time()

# ==============================================================================
# STEP 1: DATA LOADING AND PREPARATION
# ==============================================================================

cat("━"*80, "\n", sep="")
cat("STEP 1: DATA LOADING AND PREPARATION\n")
cat("━"*80, "\n\n", sep="")

# Load cleaned data
data_path <- "../../data/expenses_cleaned.csv"
df <- read.csv(data_path)

cat(sprintf("✓ Loaded %d observations from %s\n", nrow(df), basename(data_path)))

# Prepare matrices
y <- df$charges
X <- cbind(1, as.matrix(df[, c("age", "sex", "bmi", "children", "smoker")]))
colnames(X) <- c("Intercept", "Age", "Sex", "BMI", "Children", "Smoker")

n <- nrow(X)
p <- ncol(X)

cat(sprintf("✓ Response variable: charges (n = %d)\n", n))
cat(sprintf("✓ Features: %d predictors + intercept\n", p-1))

# Descriptive statistics
cat(sprintf("\nData summary:\n"))
cat(sprintf("  Mean charges: $%.2f\n", mean(y)))
cat(sprintf("  SD charges: $%.2f\n", sd(y)))
cat(sprintf("  Range: $%.2f - $%.2f\n\n", min(y), max(y)))

# ==============================================================================
# STEP 2: GIBBS SAMPLING (BASELINE MODEL)
# ==============================================================================

cat("━"*80, "\n", sep="")
cat("STEP 2: GIBBS SAMPLING (BASELINE MODEL)\n")
cat("━"*80, "\n\n", sep="")

# Priors
prior_beta_mean <- rep(0, p)
prior_beta_precision <- diag(0.001, p)
prior_sigma2_shape <- 0.01
prior_sigma2_scale <- 0.01

# Sampling parameters
n_iter <- 50000
warmup <- 10000
n_chains <- 3

cat(sprintf("Running Gibbs sampler:\n"))
cat(sprintf("  - Iterations: %d (warmup: %d)\n", n_iter, warmup))
cat(sprintf("  - Chains: %d\n", n_chains))

set.seed(42)

gibbs_results <- lapply(1:n_chains, function(chain_id) {
  cat(sprintf("  Chain %d/%d...\n", chain_id, n_chains))
  gibbs_lm(y, X, prior_beta_mean, prior_beta_precision,
           prior_sigma2_shape, prior_sigma2_scale,
           n_iter, warmup, seed = 42 + chain_id)
})

cat("\n✓ Gibbs sampling complete\n\n")

# ==============================================================================
# STEP 3: METROPOLIS-HASTINGS ANALYSIS
# ==============================================================================

cat("━"*80, "\n", sep="")
cat("STEP 3: METROPOLIS-HASTINGS ANALYSIS\n")
cat("━"*80, "\n\n", sep="")

cat(sprintf("Running Metropolis-Hastings sampler:\n"))
cat(sprintf("  - Iterations: %d (warmup: %d)\n", n_iter, warmup))
cat(sprintf("  - Chains: %d\n", n_chains))
cat(sprintf("  - Adaptive tuning: enabled\n"))

set.seed(42)

mh_results <- lapply(1:n_chains, function(chain_id) {
  cat(sprintf("  Chain %d/%d...\n", chain_id, n_chains))
  metropolis_hastings_lm(y, X, prior_beta_mean, prior_beta_precision,
                        prior_sigma2_shape, prior_sigma2_scale,
                        n_iter, warmup, seed = 42 + chain_id)
})

cat("\n✓ Metropolis-Hastings sampling complete\n\n")

# ==============================================================================
# STEP 4: CONVERGENCE DIAGNOSTICS
# ==============================================================================

cat("━"*80, "\n", sep="")
cat("STEP 4: CONVERGENCE DIAGNOSTICS\n")
cat("━"*80, "\n\n", sep="")

# Check Gibbs convergence
cat("Gibbs Sampling Convergence:\n")
gibbs_beta_chains <- lapply(gibbs_results, function(x) x$beta)
gibbs_sigma2_chains <- lapply(gibbs_results, function(x) x$sigma2)

rhat_gibbs_beta <- sapply(1:p, function(j) {
  chains <- lapply(gibbs_beta_chains, function(x) x[, j])
  mcmc_list <- mcmc.list(lapply(chains, mcmc))
  gelman.diag(mcmc_list, autoburnin = FALSE)$psrf[1, 1]
})

for (j in 1:p) {
  status <- ifelse(rhat_gibbs_beta[j] < 1.1, "✓", "✗")
  cat(sprintf("  %s %s: R-hat = %.4f\n", status, colnames(X)[j], rhat_gibbs_beta[j]))
}

# Check MH convergence
cat("\nMetropolis-Hastings Convergence:\n")
mh_beta_chains <- lapply(mh_results, function(x) x$beta)
mh_sigma2_chains <- lapply(mh_results, function(x) x$sigma2)

rhat_mh_beta <- sapply(1:p, function(j) {
  chains <- lapply(mh_beta_chains, function(x) x[, j])
  mcmc_list <- mcmc.list(lapply(chains, mcmc))
  gelman.diag(mcmc_list, autoburnin = FALSE)$psrf[1, 1]
})

for (j in 1:p) {
  status <- ifelse(rhat_mh_beta[j] < 1.1, "✓", "✗")
  cat(sprintf("  %s %s: R-hat = %.4f\n", status, colnames(X)[j], rhat_mh_beta[j]))
}

cat("\n")

# ==============================================================================
# STEP 5: ALGORITHM COMPARISON
# ==============================================================================

cat("━"*80, "\n", sep="")
cat("STEP 5: ALGORITHM COMPARISON\n")
cat("━"*80, "\n\n", sep="")

compare_algorithms(
  gibbs_results = gibbs_results,
  mh_results = mh_results,
  param_names = colnames(X),
  output_dir = "../outputs/algorithm_comparison"
)

# ==============================================================================
# STEP 6: RESIDUAL DIAGNOSTICS
# ==============================================================================

cat("\n")
cat("━"*80, "\n", sep="")
cat("STEP 6: RESIDUAL DIAGNOSTICS\n")
cat("━"*80, "\n\n", sep="")

# Combine Gibbs chains
gibbs_beta_all <- do.call(rbind, gibbs_beta_chains)
gibbs_sigma2_all <- do.call(c, gibbs_sigma2_chains)

cat("Generating residual diagnostic plots...\n\n")

plot_residual_diagnostics(
  y_obs = y,
  X = X,
  beta_samples = gibbs_beta_all,
  sigma2_samples = gibbs_sigma2_all,
  output_path = "../outputs/gibbs_result/residual_diagnostics.png",
  model_name = "Gibbs Sampler"
)

# ==============================================================================
# STEP 7: POSTERIOR INFERENCE AND SUMMARY
# ==============================================================================

cat("\n")
cat("━"*80, "\n", sep="")
cat("STEP 7: POSTERIOR INFERENCE AND SUMMARY\n")
cat("━"*80, "\n\n", sep="")

# Posterior estimates
beta_mean <- colMeans(gibbs_beta_all)
sigma2_mean <- mean(gibbs_sigma2_all)

cat("Posterior Estimates:\n")
for (j in 1:p) {
  ci_lower <- quantile(gibbs_beta_all[, j], 0.025)
  ci_upper <- quantile(gibbs_beta_all[, j], 0.975)
  cat(sprintf("  %s: %.4f [%.4f, %.4f]\n", colnames(X)[j], beta_mean[j], ci_lower, ci_upper))
}
cat(sprintf("  σ²: %.4f\n\n", sigma2_mean))

# Model fit
y_pred <- X %*% beta_mean
r_squared <- 1 - sum((y - y_pred)^2) / sum((y - mean(y))^2)
rmse <- sqrt(mean((y - y_pred)^2))
mae <- mean(abs(y - y_pred))

cat("Model Fit Statistics:\n")
cat(sprintf("  R² = %.4f (%.1f%% variance explained)\n", r_squared, r_squared * 100))
cat(sprintf("  RMSE = $%.2f\n", rmse))
cat(sprintf("  MAE = $%.2f\n\n", mae))

# Save summary
output_dir <- "../outputs/final_summary"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Posterior estimates table
posterior_summary <- data.frame(
  Parameter = c(colnames(X), "σ²"),
  Mean = c(beta_mean, sigma2_mean),
  SD = c(apply(gibbs_beta_all, 2, sd), sd(gibbs_sigma2_all)),
  CI_2.5 = c(apply(gibbs_beta_all, 2, function(x) quantile(x, 0.025)), quantile(gibbs_sigma2_all, 0.025)),
  CI_97.5 = c(apply(gibbs_beta_all, 2, function(x) quantile(x, 0.975)), quantile(gibbs_sigma2_all, 0.975))
)

write.csv(posterior_summary, file.path(output_dir, "posterior_summary.csv"), row.names = FALSE)

# Model fit table
model_fit <- data.frame(
  Statistic = c("R²", "RMSE", "MAE", "n", "p", "Effective_n"),
  Value = c(r_squared, rmse, mae, n, p, n)
)

write.csv(model_fit, file.path(output_dir, "model_fit.csv"), row.names = FALSE)

cat(sprintf("✓ Results saved to %s\n\n", output_dir))

# ==============================================================================
# FINAL SUMMARY
# ==============================================================================

analysis_end_time <- Sys.time()
total_time <- as.numeric(difftime(analysis_end_time, analysis_start_time, units = "secs"))

cat("\n")
cat("╔"*80, "\n", sep="")
cat("║ ANALYSIS COMPLETE\n")
cat("╚"*80, "\n", sep="")
cat(sprintf("Total analysis time: %.2f seconds (%.1f minutes)\n\n", total_time, total_time/60))

cat("Summary:\n")
cat(sprintf("  ✓ Data: %d observations, %d predictors\n", n, p-1))
cat(sprintf("  ✓ Algorithms: Gibbs + Metropolis-Hastings\n"))
cat(sprintf("  ✓ Chains: %d per algorithm (%d iterations each)\n", n_chains, n_iter))
cat(sprintf("  ✓ Convergence: %s\n", 
            ifelse(all(c(rhat_gibbs_beta, rhat_mh_beta) < 1.1), 
                   "All chains converged", 
                   "Check diagnostics")))
cat(sprintf("  ✓ Model fit: R² = %.3f\n\n", r_squared))

cat("Output directories:\n")
cat("  - ../outputs/gibbs_result/\n")
cat("  - ../outputs/mh_baseline/\n")
cat("  - ../outputs/algorithm_comparison/\n")
cat("  - ../outputs/final_summary/\n\n")

cat("="*80, "\n", sep="")
cat("All results saved. Analysis pipeline complete!\n")
cat("="*80, "\n", sep="")
