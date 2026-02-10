# ==============================================================================
# Run_Full_Analysis.R
# Canonical end-to-end pipeline for the project.
#
# Data -> frequentist OLS baseline -> Gibbs + Metropolis-Hastings samplers ->
# diagnostics (trace/ACF/ESS/R-hat) -> posterior predictive checks ->
# algorithm comparison -> residual diagnostics.
# ==============================================================================

# Ensure working directory is the scripts folder
if (!is.null(sys.frame(1)$ofile)) {
  setwd(dirname(sys.frame(1)$ofile))
}

suppressPackageStartupMessages({
  library(MASS)
  library(coda)
  library(ggplot2)
})

# ------------------------------------------------------------------------------
# Source project modules
# ------------------------------------------------------------------------------
source("Gibbs_Sampling.R")
source("Metropolis_Hastings.R")
source("Convergence_Detection.R")
source("Posterior_Inference.R")
source("Algorithm_Comparison.R")
source("Residual_Diagnostics.R")

# ------------------------------------------------------------------------------
# (1) Load cleaned dataset
# ------------------------------------------------------------------------------
data_path <- "../../data/expenses_cleaned.csv"
if (!file.exists(data_path)) {
  stop("Cannot find cleaned data at: ", normalizePath(data_path, winslash = "/", mustWork = FALSE))
}

df <- read.csv(data_path)

if ("sex" %in% names(df) && !is.factor(df$sex)) df$sex <- as.factor(df$sex)
if ("smoker" %in% names(df) && !is.factor(df$smoker)) df$smoker <- as.factor(df$smoker)

# ------------------------------------------------------------------------------
# (2) Frequentist baseline: OLS + correlation matrix
# ------------------------------------------------------------------------------
ols_output_dir <- file.path("..", "outputs", "frequentist_baseline")
dir.create(ols_output_dir, recursive = TRUE, showWarnings = FALSE)

ols_fit <- lm(charges ~ age + sex + bmi + children + smoker, data = df)

sink(file.path(ols_output_dir, "ols_summary.txt"))
cat("OLS baseline: charges ~ age + sex + bmi + children + smoker\n\n")
print(summary(ols_fit))
sink()

write.csv(
  coef(summary(ols_fit)),
  file = file.path(ols_output_dir, "ols_coef_table.csv"),
  row.names = TRUE
)

# Correlation matrix on numeric covariates
cor_df <- data.frame(
  age = df$age,
  bmi = df$bmi,
  children = df$children,
  sex = if ("sex" %in% names(df)) as.numeric(df$sex) - 1 else NA,
  smoker = if ("smoker" %in% names(df)) as.numeric(df$smoker) - 1 else NA,
  charges = df$charges
)
cor_matrix <- cor(cor_df, use = "complete.obs")
write.csv(cor_matrix, file = file.path(ols_output_dir, "cor_matrix.csv"))

cat("Saved OLS summary + correlation matrix to: ", normalizePath(ols_output_dir, winslash = "/", mustWork = FALSE), "\n")

# ------------------------------------------------------------------------------
# (3) Design matrix for Bayesian samplers
# ------------------------------------------------------------------------------
X <- model.matrix(~ age + sex + bmi + children + smoker, data = df)
y <- df$charges

p <- ncol(X)
feature_names <- colnames(X)

# ------------------------------------------------------------------------------
# (4) Priors + MCMC settings
# ------------------------------------------------------------------------------
# Conjugate prior:
#   beta | sigma2 ~ N(b0, sigma2 * B0^{-1})
#   sigma2 ~ Inv-Gamma(a0, d0)

b0 <- rep(0, p)
B0 <- diag(1e-4, p)       # weak prior

a0 <- 0.01
d0 <- 0.01

n_iter <- 10000
warmup <- 2000
n_chains <- 3
seed_base <- 123

# ------------------------------------------------------------------------------
# (5) Gibbs sampler
# ------------------------------------------------------------------------------
cat("\n", strrep("=", 70), "\n", sep = "")
cat("RUNNING GIBBS SAMPLER\n")
cat(strrep("=", 70), "\n", sep = "")

start_gibbs <- Sys.time()
gibbs_results <- gibbs_lm(
  y = y,
  X = X,
  n_iter = n_iter,
  warmup = warmup,
  n_chains = n_chains,
  b0 = b0,
  B0 = B0,
  a0 = a0,
  d0 = d0,
  seed = seed_base
)
gibbs_time_total <- as.numeric(difftime(Sys.time(), start_gibbs, units = "secs"))

# Inject timing (Algorithm_Comparison.R expects time_elapsed)
for (k in seq_along(gibbs_results)) {
  gibbs_results[[k]]$time_elapsed <- gibbs_time_total / n_chains
}

beta_list_gibbs <- lapply(gibbs_results, function(ch) ch$beta)
sigma2_list_gibbs <- lapply(gibbs_results, function(ch) ch$sigma2)

for (k in seq_along(beta_list_gibbs)) colnames(beta_list_gibbs[[k]]) <- feature_names

# Trace plots
beta_trace_plot(beta_list_gibbs, "gibbs_baseline")
sigma2_trace_plot(sigma2_list_gibbs, "gibbs_baseline")

# Convergence diagnostics
acf_plot_beta(beta_list_gibbs, "gibbs_baseline")
acf_plot_sigma2(sigma2_list_gibbs, "gibbs_baseline")
ess_beta_table(beta_list_gibbs, X, "gibbs_baseline")
ess_sigma2_table(sigma2_list_gibbs, "gibbs_baseline")
rhat_beta_table(beta_list_gibbs, X, "gibbs_baseline")
rhat_sigma2_table(sigma2_list_gibbs, "gibbs_baseline")

# Posterior summaries
beta_sum_gibbs <- beta_summary_stats(beta_list_gibbs)
sigma2_sum_gibbs <- sigma2_summary_stats(sigma2_list_gibbs)

out_dir_baseline <- file.path("..", "outputs", "gibbs_baseline")
dir.create(out_dir_baseline, recursive = TRUE, showWarnings = FALSE)
write.csv(beta_sum_gibbs, file = file.path(out_dir_baseline, "gibbs_beta_summary.csv"), row.names = TRUE)
write.csv(as.data.frame(t(sigma2_sum_gibbs)), file = file.path(out_dir_baseline, "gibbs_sigma2_summary.csv"), row.names = FALSE)

# Posterior predictive checks (Gibbs)
y_rep_gibbs <- posterior_predictive(beta_list_gibbs, sigma2_list_gibbs, X)
ppc_plot(y, y_rep_gibbs, "gibbs_baseline")
PPC_density_overlay(y, y_rep_gibbs, "gibbs_baseline")
ppc_residual_plot(y, y_rep_gibbs, "gibbs_baseline")

# Residual diagnostics (Gibbs)
beta_all_gibbs <- do.call(rbind, beta_list_gibbs)
sigma2_all_gibbs <- unlist(sigma2_list_gibbs)
plot_residual_diagnostics(
  y_obs = y,
  X = X,
  beta_samples = beta_all_gibbs,
  sigma2_samples = sigma2_all_gibbs,
  output_path = file.path(out_dir_baseline, "gibbs_residual_diagnostics.png"),
  model_name = "Gibbs baseline"
)

# ------------------------------------------------------------------------------
# (6) Metropolis-Hastings sampler
# ------------------------------------------------------------------------------
cat("\n", strrep("=", 70), "\n", sep = "")
cat("RUNNING METROPOLIS-HASTINGS SAMPLER\n")
cat(strrep("=", 70), "\n", sep = "")

mh_results <- metropolis_hastings_lm(
  y = y,
  X = X,
  n_iter = n_iter,
  warmup = warmup,
  n_chains = n_chains,
  b0 = b0,
  B0_inv = B0,
  a0 = a0,
  d0 = d0,
  proposal_sd_beta = 0.1,
  proposal_sd_sigma2 = 0.1,
  seed = seed_base
)

beta_list_mh <- lapply(mh_results, function(ch) ch$beta)
sigma2_list_mh <- lapply(mh_results, function(ch) ch$sigma2)
for (k in seq_along(beta_list_mh)) colnames(beta_list_mh[[k]]) <- feature_names

beta_trace_plot_mh(beta_list_mh, "mh_baseline")
sigma2_trace_plot_mh(sigma2_list_mh,"mh_baseline")

# Convergence diagnostics
acf_plot_beta(beta_list_mh, "mh_baseline")
acf_plot_sigma2(sigma2_list_mh, "mh_baseline")
ess_beta_table(beta_list_mh, X, "mh_baseline")
ess_sigma2_table(sigma2_list_mh, "mh_baseline")
rhat_beta_table(beta_list_mh, X, "mh_baseline")
rhat_sigma2_table(sigma2_list_mh, "mh_baseline")

# Posterior summaries
beta_sum_mh <- beta_summary_stats(beta_list_mh)
sigma2_sum_mh <- sigma2_summary_stats(sigma2_list_mh)

out_dir_mh <- file.path("..", "outputs", "mh_baseline")
dir.create(out_dir_mh, recursive = TRUE, showWarnings = FALSE)
write.csv(beta_sum_mh, file = file.path(out_dir_mh, "mh_beta_summary.csv"), row.names = TRUE)
write.csv(as.data.frame(t(sigma2_sum_mh)), file = file.path(out_dir_mh, "mh_sigma2_summary.csv"), row.names = FALSE)

# Posterior predictive checks (MH)
y_rep_mh <- posterior_predictive(beta_list_mh, sigma2_list_mh, X)
ppc_plot(y, y_rep_mh, "mh_baseline")
PPC_density_overlay(y, y_rep_mh, "mh_baseline")
ppc_residual_plot(y, y_rep_mh, "mh_baseline")

# Residual diagnostics (MH)
beta_all_mh <- do.call(rbind, beta_list_mh)
sigma2_all_mh <- unlist(sigma2_list_mh)
plot_residual_diagnostics(
  y_obs = y,
  X = X,
  beta_samples = beta_all_mh,
  sigma2_samples = sigma2_all_mh,
  output_path = file.path(out_dir_mh, "mh_residual_diagnostics.png"),
  model_name = "MH baseline"
)

# ------------------------------------------------------------------------------
# (7) Algorithm comparison (time/ESS/R-hat)
# ------------------------------------------------------------------------------
compare_algorithms(
  gibbs_results = gibbs_results,
  mh_results = mh_results,
  model_name = "baseline",
  output_dir = "../outputs"
)

cat("\n Full analysis finished.\n")
cat("Outputs: ", normalizePath(out_dir_baseline, winslash = "/", mustWork = FALSE), "\n")
cat("Plots (trace/ACF/PPC): ", normalizePath("plots", winslash = "/", mustWork = FALSE), "\n")
