# ==============================================================================
# Residual Diagnostics for Bayesian Linear Regression
# ==============================================================================

library(ggplot2)
library(gridExtra)

#' Compute residuals from Bayesian regression
#'
#' @param y_obs Observed response vector
#' @param X Design matrix
#' @param beta_samples Matrix of beta posterior samples
#' @param sigma2_samples Vector of sigma2 posterior samples
#' @return List with residuals and predictions
compute_residuals <- function(y_obs, X, beta_samples, sigma2_samples) {
  # Posterior mean of beta
  beta_mean <- colMeans(beta_samples)
  
  # Posterior mean predictions
  y_pred <- X %*% beta_mean
  
  # Residuals
  residuals <- y_obs - y_pred
  
  return(list(
    residuals = as.vector(residuals),
    y_pred = as.vector(y_pred),
    beta_mean = beta_mean
  ))
}


#' Compute standardized residuals
#'
#' @param residuals Raw residuals
#' @param sigma2_samples Posterior samples of sigma^2
#' @param X Design matrix
#' @return Standardized residuals
standardized_residuals <- function(residuals, sigma2_samples, X) {
  # Posterior mean of sigma
  sigma_mean <- sqrt(mean(sigma2_samples))
  
  # Compute leverage (hat matrix diagonal)
  H <- X %*% solve(t(X) %*% X) %*% t(X)
  leverage <- diag(H)
  
  # Standardize
  std_residuals <- residuals / (sigma_mean * sqrt(1 - leverage))
  
  return(std_residuals)
}


#' Generate comprehensive residual diagnostic plots
#'
#' @param y_obs Observed responses
#' @param X Design matrix
#' @param beta_samples Posterior beta samples
#' @param sigma2_samples Posterior sigma2 samples
#' @param output_path File path for saving plot
#' @param model_name Model name for title
plot_residual_diagnostics <- function(y_obs, X, beta_samples, sigma2_samples,
                                      output_path = NULL, model_name = "Model") {
  
  # Compute residuals
  res_obj <- compute_residuals(y_obs, X, beta_samples, sigma2_samples)
  residuals <- res_obj$residuals
  y_pred <- res_obj$y_pred
  
  # Standardized residuals
  std_residuals <- standardized_residuals(residuals, sigma2_samples, X)
  
  # Open graphics device if output path specified
  if (!is.null(output_path)) {
    png(output_path, width = 1200, height = 900, res = 120)
  }
  
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 2))
  
  # ==========================================
  # Plot 1: Residuals vs Fitted
  # ==========================================
  plot(y_pred, residuals,
       xlab = "Fitted values",
       ylab = "Residuals",
       main = paste(model_name, "- Residuals vs Fitted"),
       pch = 16, col = rgb(0, 0, 1, 0.5))
  abline(h = 0, col = "red", lwd = 2, lty = 2)
  
  # Add LOWESS smoother
  lowess_fit <- lowess(y_pred, residuals)
  lines(lowess_fit, col = "blue", lwd = 2)
  
  # ==========================================
  # Plot 2: Normal Q-Q Plot
  # ==========================================
  qqnorm(std_residuals, 
         main = paste(model_name, "- Normal Q-Q"),
         pch = 16, col = rgb(0, 0, 1, 0.5))
  qqline(std_residuals, col = "red", lwd = 2, lty = 2)
  
  # ==========================================
  # Plot 3: Scale-Location (sqrt of |std residuals|)
  # ==========================================
  sqrt_std_resid <- sqrt(abs(std_residuals))
  plot(y_pred, sqrt_std_resid,
       xlab = "Fitted values",
       ylab = expression(sqrt("|Standardized residuals|")),
       main = paste(model_name, "- Scale-Location"),
       pch = 16, col = rgb(0, 0, 1, 0.5))
  
  # Add LOWESS smoother
  lowess_fit2 <- lowess(y_pred, sqrt_std_resid)
  lines(lowess_fit2, col = "blue", lwd = 2)
  
  # ==========================================
  # Plot 4: Histogram of Residuals
  # ==========================================
  hist(residuals, breaks = 30, col = "lightblue", border = "white",
       main = paste(model_name, "- Residual Histogram"),
       xlab = "Residuals",
       freq = FALSE)
  
  # Overlay normal density
  x_range <- seq(min(residuals), max(residuals), length.out = 100)
  lines(x_range, dnorm(x_range, mean = mean(residuals), sd = sd(residuals)),
        col = "red", lwd = 2, lty = 2)
  
  # Add actual kernel density
  lines(density(residuals), col = "blue", lwd = 2)
  legend("topright", legend = c("Observed", "Normal"),
         col = c("blue", "red"), lwd = 2, lty = c(1, 2))
  
  if (!is.null(output_path)) {
    dev.off()
    cat(sprintf("✓ Residual diagnostics saved to %s\n", output_path))
  }
  
  # ==========================================
  # Diagnostic Tests
  # ==========================================
  cat("\n", "="*60, "\n", sep="")
  cat("RESIDUAL DIAGNOSTIC TESTS\n")
  cat("="*60, "\n", sep="")
  
  # Shapiro-Wilk test for normality
  if (length(residuals) <= 5000) {
    sw_test <- shapiro.test(residuals)
    cat(sprintf("Shapiro-Wilk normality test: W = %.4f, p-value = %.4e\n",
                sw_test$statistic, sw_test$p.value))
    if (sw_test$p.value < 0.05) {
      cat("  → Residuals deviate from normality (p < 0.05)\n")
    } else {
      cat("  → Residuals are approximately normal (p ≥ 0.05)\n")
    }
  } else {
    cat("Shapiro-Wilk test skipped (n > 5000)\n")
  }
  
  # Durbin-Watson test for autocorrelation (if lmtest is available)
  if (requireNamespace("lmtest", quietly = TRUE)) {
    # Need a linear model object for DW test
    lm_obj <- lm(y_obs ~ X - 1)
    dw_test <- lmtest::dwtest(lm_obj)
    cat(sprintf("\nDurbin-Watson test: DW = %.4f, p-value = %.4e\n",
                dw_test$statistic, dw_test$p.value))
    if (abs(dw_test$statistic - 2) > 0.5) {
      cat("  → Evidence of autocorrelation\n")
    } else {
      cat("  → No significant autocorrelation\n")
    }
  }
  
  # Breusch-Pagan test for heteroscedasticity (if lmtest is available)
  if (requireNamespace("lmtest", quietly = TRUE)) {
    lm_obj <- lm(y_obs ~ X - 1)
    bp_test <- lmtest::bptest(lm_obj)
    cat(sprintf("\nBreusch-Pagan test: BP = %.4f, p-value = %.4e\n",
                bp_test$statistic, bp_test$p.value))
    if (bp_test$p.value < 0.05) {
      cat("  → Evidence of heteroscedasticity (p < 0.05)\n")
    } else {
      cat("  → Homoscedastic (p ≥ 0.05)\n")
    }
  }
  
  cat("="*60, "\n", sep="")
  
  return(invisible(list(
    residuals = residuals,
    std_residuals = std_residuals,
    y_pred = y_pred
  )))
}


#' Compare residuals across multiple models
#'
#' @param models_list List of model results
#' @param model_names Names of models
#' @param output_path Output file path
plot_residual_comparison <- function(models_list, model_names, output_path = NULL) {
  
  n_models <- length(models_list)
  
  if (!is.null(output_path)) {
    png(output_path, width = 1400, height = 400 * ceiling(n_models/3), res = 120)
  }
  
  par(mfrow = c(ceiling(n_models/3), min(3, n_models)), mar = c(4, 4, 3, 2))
  
  for (i in 1:n_models) {
    model <- models_list[[i]]
    
    # Q-Q plot
    qqnorm(model$residuals,
           main = paste(model_names[i], "- Q-Q Plot"),
           pch = 16, col = rgb(0, 0, 1, 0.3))
    qqline(model$residuals, col = "red", lwd = 2, lty = 2)
  }
  
  if (!is.null(output_path)) {
    dev.off()
    cat(sprintf("✓ Residual comparison saved to %s\n", output_path))
  }
}


# ==============================================================================
# Example Usage
# ==============================================================================

if (FALSE) {
  # Load data
  df <- read.csv("../../data/expenses_cleaned.csv")
  
  # Prepare matrices
  y <- df$charges
  X <- cbind(1, as.matrix(df[, c("age", "sex", "bmi", "children", "smoker")]))
  
  # Run Gibbs sampler
  source("Gibbs_Sampling.R")
  results <- gibbs_lm(y, X, n_iter = 10000, warmup = 2000, n_chains = 3)
  
  # Combine chains
  beta_samples <- do.call(rbind, lapply(results, function(x) x$beta))
  sigma2_samples <- do.call(c, lapply(results, function(x) x$sigma2))
  
  # Generate residual diagnostics
  plot_residual_diagnostics(
    y_obs = y,
    X = X,
    beta_samples = beta_samples,
    sigma2_samples = sigma2_samples,
    output_path = "../outputs/baseline_model/residual_diagnostics.png",
    model_name = "Baseline Model"
  )
}
