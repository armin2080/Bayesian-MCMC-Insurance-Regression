# ==============================================================================
# Test Installation: Verify R Environment and Packages
# ==============================================================================
# This script checks that all required R packages and dependencies are
# installed and working correctly for the Bayesian MCMC analysis.
# ==============================================================================

cat("="*80, "\n", sep="")
cat("TESTING R ENVIRONMENT FOR BAYESIAN MCMC ANALYSIS\n")
cat("="*80, "\n\n", sep="")

# ==============================================================================
# 1. R VERSION CHECK
# ==============================================================================

cat("1. R Version:\n")
cat("-"*80, "\n", sep="")
cat(sprintf("  R version: %s\n", R.version.string))
cat(sprintf("  Platform: %s\n", R.version$platform))
cat(sprintf("  Architecture: %s\n\n", R.version$arch))

# Check R version is >= 3.6.0
r_version <- as.numeric(R.version$major) + as.numeric(R.version$minor) / 10
if (r_version >= 3.6) {
  cat("  ✓ R version is sufficient (>= 3.6)\n\n")
} else {
  cat("  ✗ WARNING: R version < 3.6 may have compatibility issues\n\n")
}

# ==============================================================================
# 2. REQUIRED PACKAGES
# ==============================================================================

cat("2. Required Packages:\n")
cat("-"*80, "\n", sep="")

required_packages <- c(
  "MASS",         # For mvrnorm
  "coda",         # For MCMC diagnostics
  "ggplot2",      # For plotting
  "gridExtra",    # For multi-panel plots
  "lmtest"        # For diagnostic tests (optional)
)

all_installed <- TRUE

for (pkg in required_packages) {
  is_installed <- requireNamespace(pkg, quietly = TRUE)
  
  if (is_installed) {
    # Get version
    version <- as.character(packageVersion(pkg))
    cat(sprintf("  ✓ %s (v%s)\n", pkg, version))
  } else {
    cat(sprintf("  ✗ %s - NOT INSTALLED\n", pkg))
    all_installed <- FALSE
  }
}

cat("\n")

if (!all_installed) {
  cat("Some packages are missing. Install them with:\n")
  cat("  install.packages(c(")
  missing <- required_packages[!sapply(required_packages, function(x) requireNamespace(x, quietly = TRUE))]
  cat(paste0("\"", missing, "\"", collapse = ", "))
  cat("))\n\n")
} else {
  cat("  ✓ All required packages are installed!\n\n")
}

# ==============================================================================
# 3. PACKAGE FUNCTIONALITY TESTS
# ==============================================================================

cat("3. Package Functionality Tests:\n")
cat("-"*80, "\n", sep="")

# Test MASS::mvrnorm
tryCatch({
  library(MASS)
  test_mvrnorm <- mvrnorm(10, mu = c(0, 0), Sigma = diag(2))
  cat("  ✓ MASS::mvrnorm works\n")
}, error = function(e) {
  cat(sprintf("  ✗ MASS::mvrnorm failed: %s\n", e$message))
})

# Test coda functions
tryCatch({
  library(coda)
  test_chain <- mcmc(rnorm(100))
  test_ess <- effectiveSize(test_chain)
  test_chains <- mcmc.list(test_chain, test_chain)
  test_gelman <- gelman.diag(test_chains)
  cat("  ✓ coda diagnostics work (ESS, R-hat)\n")
}, error = function(e) {
  cat(sprintf("  ✗ coda diagnostics failed: %s\n", e$message))
})

# Test ggplot2
tryCatch({
  library(ggplot2)
  test_plot <- ggplot(data.frame(x = 1:10, y = 1:10), aes(x, y)) + geom_point()
  cat("  ✓ ggplot2 works\n")
}, error = function(e) {
  cat(sprintf("  ✗ ggplot2 failed: %s\n", e$message))
})

# Test gridExtra
tryCatch({
  library(gridExtra)
  cat("  ✓ gridExtra loaded\n")
}, error = function(e) {
  cat(sprintf("  ✗ gridExtra failed: %s\n", e$message))
})

# Test lmtest (optional)
tryCatch({
  library(lmtest)
  test_lm <- lm(rnorm(20) ~ rnorm(20))
  test_dw <- dwtest(test_lm)
  cat("  ✓ lmtest works (diagnostic tests available)\n")
}, error = function(e) {
  cat("  ⚠ lmtest not available (optional, used for residual diagnostics)\n")
})

cat("\n")

# ==============================================================================
# 4. DATA FILE CHECK
# ==============================================================================

cat("4. Data Files:\n")
cat("-"*80, "\n", sep="")

data_files <- c(
  "../../data/expenses.csv",
  "../../data/expenses_cleaned.csv"
)

for (file in data_files) {
  if (file.exists(file)) {
    file_info <- file.info(file)
    cat(sprintf("  ✓ %s (%d KB)\n", basename(file), round(file_info$size / 1024)))
  } else {
    cat(sprintf("  ✗ %s - NOT FOUND\n", basename(file)))
  }
}

cat("\n")

# ==============================================================================
# 5. SCRIPT FILES CHECK
# ==============================================================================

cat("5. Analysis Scripts:\n")
cat("-"*80, "\n", sep="")

script_files <- c(
  "Data_Preprocessing.R",
  "Model_Setup.R",
  "Gibbs_Sampling.R",
  "Metropolis_Hastings.R",
  "Convergence_Detection.R",
  "Posterior_Inference.R",
  "Algorithm_Comparison.R",
  "Residual_Diagnostics.R"
)

all_scripts_exist <- TRUE

for (script in script_files) {
  if (file.exists(script)) {
    cat(sprintf("  ✓ %s\n", script))
  } else {
    cat(sprintf("  ✗ %s - NOT FOUND\n", script))
    all_scripts_exist <- FALSE
  }
}

cat("\n")

# ==============================================================================
# 6. SMALL FUNCTIONALITY TEST
# ==============================================================================

cat("6. Quick Functionality Test:\n")
cat("-"*80, "\n", sep="")

tryCatch({
  # Generate test data
  set.seed(42)
  n <- 100
  p <- 3
  X <- cbind(1, matrix(rnorm(n * (p-1)), n, p-1))
  beta_true <- c(2, 1.5, -1)
  sigma2_true <- 1
  y <- X %*% beta_true + rnorm(n, 0, sqrt(sigma2_true))
  
  # Simple Gibbs sampling (10 iterations)
  library(MASS)
  
  # Prior
  prior_beta_mean <- rep(0, p)
  prior_beta_precision <- diag(0.001, p)
  prior_sigma2_shape <- 0.01
  prior_sigma2_scale <- 0.01
  
  # One Gibbs iteration
  n_iter <- 10
  beta_samples <- matrix(NA, n_iter, p)
  sigma2_samples <- numeric(n_iter)
  
  # Initialize
  beta <- rep(0, p)
  sigma2 <- 1
  
  for (i in 1:n_iter) {
    # Sample beta | sigma2, y
    V_beta <- solve(prior_beta_precision + (1/sigma2) * t(X) %*% X)
    m_beta <- V_beta %*% (prior_beta_precision %*% prior_beta_mean + (1/sigma2) * t(X) %*% y)
    beta <- mvrnorm(1, m_beta, V_beta)
    
    # Sample sigma2 | beta, y
    residuals <- y - X %*% beta
    shape_post <- prior_sigma2_shape + n/2
    scale_post <- prior_sigma2_scale + sum(residuals^2)/2
    sigma2 <- 1 / rgamma(1, shape_post, rate = scale_post)
    
    beta_samples[i, ] <- beta
    sigma2_samples[i] <- sigma2
  }
  
  cat("  ✓ Gibbs sampler test successful\n")
  cat(sprintf("    - Generated %d samples\n", n_iter))
  cat(sprintf("    - Final beta estimate: (%.2f, %.2f, %.2f)\n", 
              beta[1], beta[2], beta[3]))
  cat(sprintf("    - True beta: (%.2f, %.2f, %.2f)\n", 
              beta_true[1], beta_true[2], beta_true[3]))
  
}, error = function(e) {
  cat(sprintf("  ✗ Gibbs sampler test failed: %s\n", e$message))
})

cat("\n")

# ==============================================================================
# SUMMARY
# ==============================================================================

cat("="*80, "\n", sep="")
cat("INSTALLATION TEST SUMMARY\n")
cat("="*80, "\n", sep="")

if (all_installed && all_scripts_exist) {
  cat("✓ Environment is ready for Bayesian MCMC analysis!\n")
  cat("\nYou can now run:\n")
  cat("  - Quick_Analysis.R for a fast baseline analysis\n")
  cat("  - Run_Full_Analysis.R for complete analysis with all diagnostics\n")
} else {
  cat("✗ Some components are missing. Please install required packages and scripts.\n")
}

cat("="*80, "\n", sep="")
