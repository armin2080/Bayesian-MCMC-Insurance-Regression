# ==============================================================================
# Model_Setup_Final.R
# Purpose: Run baseline / log / interaction model variants and produce a compact
#          summary table (runtime, Avg ESS, ESS/s, MH acceptance)
#
# This script is intentionally Monte-Carlo focused: it times the samplers and
# summarizes mixing/efficiency metrics.
# ==============================================================================

# Ensure working directory is the scripts folder
if (!is.null(sys.frame(1)$ofile)) {
  setwd(dirname(sys.frame(1)$ofile))
}

suppressPackageStartupMessages({
  library(MASS)
  library(coda)
  library(dplyr)
  library(readr)
  library(stringr)
})

# ------------------------------------------------------------------------------
# Source project modules (expected to be in the same folder as this script)
# ------------------------------------------------------------------------------
source("Gibbs_Sampling.R")
source("Metropolis_Hastings.R")
source("Convergence_Detection.R")

# ------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------
n_iter   <- 10000
warmup   <- 2000
n_chains <- 4
seed_base <- 123

# Priors (match main pipeline defaults)
# Normal-Inverse-Gamma: beta | sigma2 ~ N(b0, sigma2 * B0^{-1}), sigma2 ~ IG(a0, d0)
# Here B0 is prior precision (Lambda0 in report). In code we pass B0_inv = B0 to MH.
a0 <- 0.01
d0 <- 0.01

# ------------------------------------------------------------------------------
# Data
# ------------------------------------------------------------------------------
data_path <- file.path("..", "data", "expenses_cleaned.csv")
if (!file.exists(data_path)) {
  stop("Could not find cleaned data at: ", data_path,
       "\nRun Data_Preprocessing.R first, or place expenses_cleaned.csv in ../data/.")
}
expenses_clean <- read_csv(data_path, show_col_types = FALSE)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------

# Compute capped ESS for a multi-chain matrix list (matches ess_beta_table cap logic)
ess_capped_beta <- function(beta_list) {
  mlist <- coda::mcmc.list(lapply(beta_list, coda::mcmc))
  ess_vals <- coda::effectiveSize(mlist)
  Ntot <- nrow(beta_list[[1]]) * length(beta_list)
  pmin(as.numeric(ess_vals), Ntot)
}

ess_capped_sigma2 <- function(sigma2_list) {
  mlist <- coda::mcmc.list(lapply(sigma2_list, coda::mcmc))
  ess_val <- as.numeric(coda::effectiveSize(mlist))
  Ntot <- length(sigma2_list[[1]]) * length(sigma2_list)
  min(ess_val, Ntot)
}

# Extract acceptance rates (in %) from MH output and average across chains
mh_acceptance_avg <- function(mh_results) {
  acc_beta <- mean(sapply(mh_results, function(ch) ch$acceptance_rate_beta)) * 100
  acc_sig  <- mean(sapply(mh_results, function(ch) ch$acceptance_rate_sigma2)) * 100
  c(acc_beta = acc_beta, acc_sigma2 = acc_sig)
}

run_variant <- function(model_label, y_vec, X_mat, feature_names) {

  p <- ncol(X_mat)
  b0 <- rep(0, p)
  B0 <- diag(1e-4, p)  # weak prior precision

  # ---------------------------
  # Gibbs
  # ---------------------------
  set.seed(seed_base)
  t_gibbs <- system.time({
    gibbs_res <- gibbs_lm(
      y = y_vec,
      X = X_mat,
      n_iter = n_iter,
      warmup = warmup,
      n_chains = n_chains,
      b0 = b0,
      B0 = B0,
      a0 = a0,
      d0 = d0
    )
  })

  beta_g <- lapply(gibbs_res, function(ch) ch$beta)
  sigma2_g <- lapply(gibbs_res, function(ch) ch$sigma2)
  for (k in seq_along(beta_g)) colnames(beta_g[[k]]) <- feature_names

  ess_beta_g <- ess_capped_beta(beta_g)
  ess_sig_g  <- ess_capped_sigma2(sigma2_g)
  avg_ess_g  <- mean(c(ess_beta_g, ess_sig_g))


  # ---------------------------
  # Metropolis-Hastings
  # ---------------------------
  set.seed(seed_base)
  t_mh <- system.time({
    mh_res <- metropolis_hastings_lm(
      y = y_vec,
      X = X_mat,
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
  })

  beta_mh <- lapply(mh_res, function(ch) ch$beta)
  sigma2_mh <- lapply(mh_res, function(ch) ch$sigma2)
  for (k in seq_along(beta_mh)) colnames(beta_mh[[k]]) <- feature_names

  ess_beta_mh <- ess_capped_beta(beta_mh)
  ess_sig_mh  <- ess_capped_sigma2(sigma2_mh)
  avg_ess_mh  <- mean(c(ess_beta_mh, ess_sig_mh))

  acc <- mh_acceptance_avg(mh_res)

  # Build rows
  tibble(
    Model = model_label,
    Sampler = c("Gibbs", "Metropolis--Hastings"),
    Runtime_s = c(unname(t_gibbs["elapsed"]), unname(t_mh["elapsed"])),
    Avg_ESS = c(avg_ess_g, avg_ess_mh),
    ESS_per_s = c(avg_ess_g / unname(t_gibbs["elapsed"]), avg_ess_mh / unname(t_mh["elapsed"])),
    Acc_beta_pct = c(NA_real_, acc["acc_beta"]),
    Acc_sigma2_pct = c(NA_real_, acc["acc_sigma2"])
  )
}

# ------------------------------------------------------------------------------
# Define variants
# ------------------------------------------------------------------------------
# Baseline
y_base <- expenses_clean$charges
X_base <- model.matrix(~ age + sex + bmi + children + smoker, data = expenses_clean)
feat_base <- colnames(X_base)

# Log model (log charges)
y_log <- log(expenses_clean$charges)
X_log <- X_base
feat_log <- feat_base

# Interaction model (log charges with smoker:bmi)
X_int <- model.matrix(~ age + sex + bmi + children + smoker + smoker:bmi, data = expenses_clean)
feat_int <- colnames(X_int)
y_int <- y_log

# ------------------------------------------------------------------------------
# Run all variants and create summary
# ------------------------------------------------------------------------------
variant_summary <- bind_rows(
  run_variant("Baseline (charges)", y_base, X_base, feat_base),
  run_variant("Log model (log(charges))", y_log, X_log, feat_log),
  run_variant("Interaction model", y_int, X_int, feat_int)
)

out_dir <- file.path("..", "outputs", "model_variants")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
write_csv(variant_summary, file.path(out_dir, "variant_mcmc_summary.csv"))

