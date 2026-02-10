
library(MASS)

#' @param y Numeric vector. Response variable.
#' @param X Numeric matrix or data frame. Predictor variables.
#' @param n_iter Integer. Total number of Gibbs sampling iterations. Default is 10000.
#' @param warmup Integer. Number of initial samples to discard (burn-in). Default is 2000.
#' @param n_chains Integer. Number of independent MCMC chains to run. Default is 3.
#' @param b0 Numeric vector. Prior mean for regression coefficients (\eqn{\beta}). Default is a zero vector.
#' @param B0 Numeric matrix. Prior precision matrix for regression coefficients (\eqn{\beta}). 
#'   Default is a diagonal matrix with small values (1e-4) indicating weak prior information.
#' @param a0 Numeric. Shape parameter for inverse gamma prior on \eqn{\sigma^2}. Default is 0.01.
#' @param d0 Numeric. Scale parameter for inverse gamma prior on \eqn{\sigma^2}. Default is 0.01.
#' @param seed Integer. Random seed base value for reproducibility across chains. Default is 123.
#
#' @return A list containing posterior samples from each chain:
#' \describe{
#'   \item{chain_results}{A list of length equal to `n_chains`, where each element contains:}
#'     \describe{
#'       \item{beta}{Matrix of sampled regression coefficients after burn-in; rows correspond to iterations, columns to predictors.}
#'       \item{sigma2}{Vector of sampled error variances after burn-in for that chain.}
#'     }
#' }
#'

gibbs_lm <- function(y, X,
                     n_iter = 10000,
                     warmup = 2000,
                     n_chains = 3,
                     b0 = NULL,
                     B0 = NULL,
                     a0 = 0.01,
                     d0 = 0.01,
                     seed = 123) {
  
  

  
  n <- length(y) # number of observations
  p <- ncol(X)   # number of predictors 
  
  

  # -----------------------------
  # Default priors
  # -----------------------------
  
  if (is.null(b0)) b0 <- rep(0, p)
  if (is.null(B0)) B0 <- diag(1e-4, p)
  
  # Initialize chain results
  chain_results <- vector("list", n_chains)  
  
  
  for (chain in seq_len(n_chains)) {
    
  set.seed(seed + chain)
    
  # Storage for samples
  beta_store   <- matrix(NA, n_iter, p)
  sigma2_store <- numeric(n_iter)
  
  

  # Initialize parameters
  beta_draw  <- rep(0, p)
  sigma2_draw <- 1
  
  
  # Pre-compute for efficiency
  XtX <- t(X) %*% X
  Xty <- t(X) %*% y
  

  # Gibbs sampling loop
  
  Bn <- XtX + B0         
  bn <- solve(Bn, Xty + B0 %*% b0)
  
  for (iter in 1:n_iter) {
    
    # ----- Sample beta given sigma^2 -----
    
    beta_draw <- mvrnorm(
      n = 1,
      mu = bn,
      Sigma = sigma2_draw * solve(Bn)
    )
    
    # ----- Sample sigma^2 given beta -----
    
    resid <- y - X %*% beta_draw
    a_n <- a0 + n / 2
    
    quad_prior <- t(beta_draw - b0) %*% B0 %*% (beta_draw - b0)
    
    d_n <- d0 + 0.5 * (sum(resid^2) + as.numeric(quad_prior))
    
    sigma2_draw <- 1 / rgamma(1, shape = a_n, rate = d_n)
    
    # --- ------ Store samples -----------
    
    beta_store[iter, ]  <- beta_draw
    sigma2_store[iter]  <- sigma2_draw
  }
  
  # --------------------------------------
  # Return posterior samples after burn-in
  # --------------------------------------
  
  chain_results[[chain]] <- list(
    beta   = beta_store[-(1:warmup), ],
    sigma2 = sigma2_store[-(1:warmup)]
  )
  
  cat("Chain", chain, "completed.\n")
  }
  
  return(chain_results)
}

#-----------------------------------------
# Trace Plots for Beta & Sigma^2
#-----------------------------------------

beta_trace_plot <- function(beta_list, model_name) {
  
  n_chains <- length(beta_list)
  p <- ncol(beta_list[[1]])

  plot_dir <- file.path("..", "outputs", model_name, "plots", "trace")
  
  if (!dir.exists(plot_dir)) {
    dir.create(plot_dir, recursive = TRUE)
    message("Created folder: ", plot_dir)
  }
  
  colors <- rainbow(n_chains)
  
  for (j in 1:p) {
    
    filename <- file.path(plot_dir, paste0("beta_trace_", j, ".png"))
    png(filename = filename, width = 1600, height = 1200, res = 150)
    plot(NULL,
         xlim = c(1, nrow(beta_list[[1]])),
         ylim = range(sapply(beta_list, function(x) x[, j])),
         type = "n",
         main = bquote("Trace plot: " ~ beta[.(j)]),
         xlab = "Iteration",
         ylab = bquote(beta[.(j)]))
    
    for (chain in 1:n_chains) {
      lines(beta_list[[chain]][, j], col = colors[chain])
    }
    legend("topright", legend = paste("Chain", 1:n_chains),
           col = colors, lty = 1, cex = 0.8)
    
    dev.off()
    message("Saved: ", filename)
  }
  
}


sigma2_trace_plot <- function(sigma2_list,model_name){
  
  n_chains <- length(sigma2_list)
  colors <- rainbow(n_chains)
  
  plot_dir <- file.path("..", "outputs", model_name, "plots", "trace")
  
  if (!dir.exists(plot_dir)) {
    dir.create(plot_dir, recursive = TRUE)
    message("Created folder: ", plot_dir)
  }
  
  filename <- file.path(plot_dir, paste0("Sigma_trace.png"))
  png(filename = filename, width = 1600, height = 1200, res = 150)
  plot(NULL,
       xlim = c(1, length(sigma2_list[[1]])),
       ylim = range(sapply(sigma2_list, range)),
       type = "n",
       main = expression("Trace plot: " ~ sigma^2),
       xlab = "Iteration",
       ylab = expression(sigma^2))
  
  # Plot each chain
  for (chain in 1:n_chains) {
    lines(sigma2_list[[chain]], col = colors[chain], lwd = 1.5)
  }
  
  legend("topright", legend = paste("Chain", 1:n_chains),
         col = colors, lty = 1, cex = 0.8)
  dev.off()
  message("Saved: ", filename)
}

#---------------------------------------------
# Posterior Summary Statistics
#---------------------------------------------

beta_summary_stats <- function(beta_list) {
  
  #Combine all chains
  beta_all <- do.call(rbind, beta_list)
  beta_summary <- apply(beta_all, 2, function(x) {
    c(
      mean = mean(x),
      sd   = sd(x),
      q2.5 = quantile(x, 0.025),
      q97.5 = quantile(x, 0.975)
    )
  })
  
  # Transpose + add column names
  beta_summary <- t(beta_summary)
  colnames(beta_summary) <- c("Mean", "SD", "2.5%", "97.5%")
  
  return(round(beta_summary, 4))
}


sigma2_summary_stats <- function(sigma2_list) {
  
  sigma2_all <- unlist(sigma2_list)
  sigma2_summary <- c(
    mean = mean(sigma2_all),
    sd   = sd(sigma2_all),
    q2.5 = quantile(sigma2_all, 0.025),
    q97.5 = quantile(sigma2_all, 0.975)
  )
  
  return(round(sigma2_summary, 4))
}

