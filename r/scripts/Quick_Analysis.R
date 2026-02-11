# ==============================================================================
# Quick_Analysis.R
# Fast sanity-check run (Gibbs only) producing a minimal set of outputs.
# ==============================================================================

# Ensure working directory = this script's directory
if (!is.null(sys.frame(1)$ofile)) {
  setwd(dirname(sys.frame(1)$ofile))
}

suppressPackageStartupMessages({
  library(MASS)
  library(coda)
  library(ggplot2)
})

source("Gibbs_Sampling.R")
source("Convergence_Detection.R")
source("Posterior_Inference.R")

data_path <- "../../data/expenses_cleaned.csv"
if (!file.exists(data_path)) {
  stop(sprintf("Missing data file: %s", data_path))
}

df <- read.csv(data_path)

if ("sex" %in% names(df)) df$sex <- as.factor(df$sex)
if ("smoker" %in% names(df)) df$smoker <- as.factor(df$smoker)

ols_formula <- charges ~ age + sex + bmi + children + smoker
X <- model.matrix(ols_formula, data = df)
y <- df$charges

# Quick MCMC settings
n_iter <- 5000
warmup <- 1000
n_chains <- 2
seed_base <- 123

# Priors
p <- ncol(X)
b0 <- rep(0, p)
B0 <- diag(1e-4, p)
a0 <- 0.01
d0 <- 0.01

out_dir <- file.path("../outputs", "quick_gibbs")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

cat("Running quick Gibbs sampler...\n")
res <- gibbs_lm(
  y = y, X = X,
  n_iter = n_iter, warmup = warmup, n_chains = n_chains,
  b0 = b0, B0 = B0,
  a0 = a0, d0 = d0,
  seed = seed_base
)

beta_list <- lapply(res, function(ch) ch$beta)
sigma2_list <- lapply(res, function(ch) ch$sigma2)

for (k in seq_along(beta_list)) colnames(beta_list[[k]]) <- colnames(X)

# Summaries
beta_sum <- beta_summary_stats(beta_list)
sigma2_sum <- sigma2_summary_stats(sigma2_list)

write.csv(beta_sum, file = file.path(out_dir, "beta_summary.csv"), row.names = TRUE)
write.csv(as.data.frame(t(sigma2_sum)), file = file.path(out_dir, "sigma2_summary.csv"), row.names = FALSE)

# Minimal plots
beta_trace_plot(beta_list, model_name = "quick_gibbs")
sigma2_trace_plot(sigma2_list, model_name = "quick_gibbs")

yrep <- posterior_predictive(beta_list, sigma2_list, X)
PPC_density_overlay(y, yrep, model_name = "quick_gibbs")

# Diagnostics tables (saved by the functions)
ess_beta_table(beta_list, X, model_name = "quick_gibbs")
ess_sigma2_table(sigma2_list, model_name = "quick_gibbs")
rhat_beta_table(beta_list, X, model_name = "quick_gibbs")
rhat_sigma2_table(sigma2_list, model_name = "quick_gibbs")

cat("Done. Outputs in:", normalizePath(out_dir, winslash = "/"), "\n")
