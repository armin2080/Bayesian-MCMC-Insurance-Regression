# ==============================================================================
# Quick Analysis: Fast Bayesian Regression with Gibbs Sampling
# ==============================================================================
# This script performs a quick Bayesian regression analysis on the insurance
# data using Gibbs sampling with conjugate priors.
# ==============================================================================

# Set up paths
setwd(dirname(sys.frame(1)$ofile))  # Set working directory to script location

# Load required libraries
suppressPackageStartupMessages({
  library(coda)
})

# Source necessary scripts
source("Data_Preprocessing.R")
source("Gibbs_Sampling.R")
source("Model_Setup.R")
source("Convergence_Detection.R")
source("Posterior_Inference.R")

# ==============================================================================
# 1. LOAD AND PREPARE DATA
# ==============================================================================

cat("="*80, "\n", sep="")
cat("QUICK BAYESIAN REGRESSION ANALYSIS\n")
cat("="*80, "\n\n", sep="")

cat("Step 1: Loading and preprocessing data...\n")

# Load cleaned data
data_path <- "../../data/expenses_cleaned.csv"
df <- read.csv(data_path)

cat(sprintf("  ✓ Loaded %d observations\n", nrow(df)))
cat(sprintf("  ✓ Features: %s\n", paste(setdiff(colnames(df), "charges"), collapse=", ")))

# Prepare matrices
y <- df$charges
X <- cbind(1, as.matrix(df[, c("age", "sex", "bmi", "children", "smoker")]))
colnames(X) <- c("Intercept", "Age", "Sex", "BMI", "Children", "Smoker")

n <- nrow(X)
p <- ncol(X)

cat(sprintf("\n  → Response: charges (n = %d)\n", n))
cat(sprintf("  → Predictors: %d parameters (including intercept)\n", p))

# ==============================================================================
# 2. SET UP PRIORS
# ==============================================================================

cat("\nStep 2: Setting up conjugate priors...\n")

# Weakly informative priors
prior_beta_mean <- rep(0, p)
prior_beta_precision <- diag(0.001, p)  # Large prior variance (weak prior)

prior_sigma2_shape <- 0.01
prior_sigma2_scale <- 0.01

cat("  ✓ Beta prior: Normal(0, 1000*I)\n")
cat(sprintf("  ✓ Sigma² prior: Inverse-Gamma(%.2f, %.2f)\n\n", 
            prior_sigma2_shape, prior_sigma2_scale))

# ==============================================================================
# 3. RUN GIBBS SAMPLER
# ==============================================================================

cat("Step 3: Running Gibbs sampler...\n")

set.seed(42)  # For reproducibility

# Sampling parameters
n_iter <- 50000
warmup <- 10000
n_chains <- 3

cat(sprintf("  - Iterations: %d (warmup: %d)\n", n_iter, warmup))
cat(sprintf("  - Chains: %d\n", n_chains))
cat(sprintf("  - Thinning: 1 (no thinning)\n\n"))

# Run sampler
start_time <- Sys.time()

results <- lapply(1:n_chains, function(chain_id) {
  cat(sprintf("  Running chain %d/%d...\n", chain_id, n_chains))
  
  gibbs_lm(
    y = y,
    X = X,
    prior_beta_mean = prior_beta_mean,
    prior_beta_precision = prior_beta_precision,
    prior_sigma2_shape = prior_sigma2_shape,
    prior_sigma2_scale = prior_sigma2_scale,
    n_iter = n_iter,
    warmup = warmup,
    seed = 42 + chain_id
  )
})

end_time <- Sys.time()
elapsed_time <- as.numeric(difftime(end_time, start_time, units = "secs"))

cat(sprintf("\n  ✓ Sampling completed in %.2f seconds\n", elapsed_time))

# ==============================================================================
# 4. CONVERGENCE DIAGNOSTICS
# ==============================================================================

cat("\nStep 4: Checking convergence...\n")

# Extract chains
beta_chains <- lapply(results, function(x) x$beta)
sigma2_chains <- lapply(results, function(x) x$sigma2)

# Compute R-hat for all parameters
rhat_beta <- sapply(1:p, function(j) {
  chains <- lapply(beta_chains, function(x) x[, j])
  mcmc_list <- mcmc.list(lapply(chains, mcmc))
  gelman.diag(mcmc_list, autoburnin = FALSE)$psrf[1, 1]
})

sigma2_mcmc_list <- mcmc.list(lapply(sigma2_chains, mcmc))
rhat_sigma2 <- gelman.diag(sigma2_mcmc_list, autoburnin = FALSE)$psrf[1, 1]

# Compute ESS
ess_beta <- sapply(1:p, function(j) {
  chains <- lapply(beta_chains, function(x) x[, j])
  mcmc_list <- mcmc.list(lapply(chains, mcmc))
  min(effectiveSize(mcmc_list))
})

ess_sigma2 <- min(effectiveSize(sigma2_mcmc_list))

# Print convergence results
cat("\n  R-hat values (should be < 1.1):\n")
for (j in 1:p) {
  status <- ifelse(rhat_beta[j] < 1.1, "✓", "✗")
  cat(sprintf("    %s %s: %.4f\n", status, colnames(X)[j], rhat_beta[j]))
}
cat(sprintf("    %s σ²: %.4f\n", ifelse(rhat_sigma2 < 1.1, "✓", "✗"), rhat_sigma2))

# Check if all converged
all_converged <- all(c(rhat_beta, rhat_sigma2) < 1.1)

if (all_converged) {
  cat("\n  ✓ ALL CHAINS CONVERGED!\n")
} else {
  cat("\n  ✗ WARNING: Some chains did not converge!\n")
}

# ==============================================================================
# 5. POSTERIOR ESTIMATES
# ==============================================================================

cat("\nStep 5: Computing posterior estimates...\n\n")

# Combine chains
beta_all <- do.call(rbind, beta_chains)
sigma2_all <- do.call(c, sigma2_chains)

# Calculate posterior means and credible intervals
posterior_summary <- data.frame()

for (j in 1:p) {
  samples <- beta_all[, j]
  posterior_summary <- rbind(posterior_summary, data.frame(
    Parameter = colnames(X)[j],
    Mean = mean(samples),
    SD = sd(samples),
    CI_2.5 = quantile(samples, 0.025),
    CI_97.5 = quantile(samples, 0.975),
    ESS = ess_beta[j],
    Rhat = rhat_beta[j]
  ))
}

# Add sigma²
posterior_summary <- rbind(posterior_summary, data.frame(
  Parameter = "σ²",
  Mean = mean(sigma2_all),
  SD = sd(sigma2_all),
  CI_2.5 = quantile(sigma2_all, 0.025),
  CI_97.5 = quantile(sigma2_all, 0.975),
  ESS = ess_sigma2,
  Rhat = rhat_sigma2
))

cat("POSTERIOR ESTIMATES (95% Credible Intervals):\n")
cat("="*80, "\n", sep="")
print(posterior_summary, row.names = FALSE, digits = 4)

# ==============================================================================
# 6. MODEL FIT STATISTICS
# ==============================================================================

cat("\n\nStep 6: Computing model fit statistics...\n")

# Posterior predictive mean
beta_mean <- colMeans(beta_all)
y_pred <- X %*% beta_mean

# R-squared
ss_res <- sum((y - y_pred)^2)
ss_tot <- sum((y - mean(y))^2)
r_squared <- 1 - ss_res / ss_tot

# RMSE
rmse <- sqrt(mean((y - y_pred)^2))

# MAE
mae <- mean(abs(y - y_pred))

cat(sprintf("\n  R² = %.4f (%.1f%% variance explained)\n", r_squared, r_squared * 100))
cat(sprintf("  RMSE = $%.2f\n", rmse))
cat(sprintf("  MAE = $%.2f\n", mae))

# ==============================================================================
# 7. SAVE RESULTS
# ==============================================================================

cat("\nStep 7: Saving results...\n")

# Create output directory
output_dir <- "../outputs/quick_analysis"
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Save posterior estimates
write.csv(posterior_summary, file.path(output_dir, "posterior_estimates.csv"), 
          row.names = FALSE)
cat(sprintf("  ✓ Posterior estimates saved to %s/posterior_estimates.csv\n", output_dir))

# Save model fit statistics
model_stats <- data.frame(
  Statistic = c("R²", "RMSE", "MAE", "n", "p"),
  Value = c(r_squared, rmse, mae, n, p)
)
write.csv(model_stats, file.path(output_dir, "model_fit.csv"), row.names = FALSE)
cat(sprintf("  ✓ Model fit statistics saved to %s/model_fit.csv\n", output_dir))

# ==============================================================================
# SUMMARY
# ==============================================================================

cat("\n")
cat("="*80, "\n", sep="")
cat("ANALYSIS COMPLETE\n")
cat("="*80, "\n", sep="")
cat(sprintf("Total time: %.2f seconds\n", elapsed_time))
cat(sprintf("Convergence: %s\n", ifelse(all_converged, "✓ All chains converged", "✗ Check diagnostics")))
cat(sprintf("Model fit: R² = %.3f, RMSE = $%.0f\n", r_squared, rmse))
cat(sprintf("Output directory: %s\n", output_dir))
cat("="*80, "\n", sep="")
