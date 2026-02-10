#======================================================
# (4) Convergence Detection
#     - Autocorrelation Function (ACF)
#     - Effective Sample Size (ESS)
#======================================================



#------------------------------------------------------
# (4.1) Autocorrelation Function (ACF) Plots
#------------------------------------------------------
# These functions generate ACF plots for MCMC samples
# and save them for convergence assessment.
#------------------------------------------------------


#------------------------------------------------------
# ACF Plot for regression coefficients (beta)
#------------------------------------------------------
# Args:
#   beta_list  : list of MCMC chains, each a matrix (iterations * p)
#   model_name : character string used to name output
#   lag_max    : maximum lag for ACF computation
#------------------------------------------------------

acf_plot_beta <- function(beta_list, model_name, lag_max = 50, ylim_zoom = 0.1){
  
  # create output directory
  plot_dir <- file.path("..", "outputs", model_name, "plots", "ACF_plots")
  full_dir <- file.path(plot_dir, "full")
  zoom_dir <- file.path(plot_dir, "zoomed")
  
  dir.create(full_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(zoom_dir, recursive = TRUE, showWarnings = FALSE)
  
  # number of regression coefficients
  p <- ncol(beta_list[[1]])
  
  # loop of coefficients
  for (j in 1:p) {
    
    # combine all chains for coefficient j
    beta_samples <- do.call(c, lapply(beta_list, function(x) x[, j]))
    
    
    #----------------------------
    # Full ACF (with lag 0)
    #----------------------------
    
    full_file <- file.path(full_dir, paste0("acf_beta_", j-1, "_full.png"))
    png(full_file, width = 1600, height = 1200, res = 150)
    
    acf(
      beta_samples,
      lag.max = lag_max,
      main = bquote("Full ACF of " ~ beta[.(j-1)])
    )
    
    dev.off()
    message("Saved: ", full_file)
    
    
    #----------------------------
    # Zoomed ACF (exclude lag 0)
    #----------------------------
    
    acf_obj <- acf(beta_samples, lag.max = lag_max, plot = FALSE)
    
    zoom_file <- file.path(zoom_dir, paste0("acf_beta_", j-1, "_zoomed.png"))
    png(zoom_file, width = 1600, height = 1200, res = 150)
    
    plot(
      acf_obj$lag[-1],
      acf_obj$acf[-1],
      type = "h",
      ylim = c(-ylim_zoom, ylim_zoom),
      xlab = "Lag",
      ylab = "ACF",
      main = bquote("Zoomed ACF of " ~ beta[.(j-1)])
    )
    
    ci <- 1.96 / sqrt(length(beta_samples))
    abline(h = c(-ci, ci), col = "blue", lty = 2)
    abline(h = 0, col = "red")
    
    dev.off()
    message("Saved: ", zoom_file)
  }
}


#------------------------------------------------------
# ACF Plot for error variance (sigma^2)
#------------------------------------------------------
# Args:
#   sigma2_list : list of MCMC chains for sigma^2
#   model_name  : character string used to name output
#   lag_max     : maximum lag for ACF computation
#------------------------------------------------------

acf_plot_sigma2 <- function(sigma2_list, model_name, lag_max = 50, ylim_zoom = 0.1) {
  
  plot_dir <- file.path("..", "outputs", model_name, "plots", "ACF_plots")
  full_dir <- file.path(plot_dir, "full")
  zoom_dir <- file.path(plot_dir, "zoomed")

  dir.create(full_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(zoom_dir, recursive = TRUE, showWarnings = FALSE)
  
  sigma2_samples <- unlist(sigma2_list)
  
  
  #---------------------------------
  # Full ACF
  #---------------------------------
  
  full_file <- file.path(full_dir, "acf_sigma2_full.png")
  png(full_file, width = 1600, height = 1200, res = 150)
  
  acf(
    sigma2_samples,
    lag.max = lag_max,
    main = expression("Full ACF of " ~ sigma^2)
  )
  dev.off()
  message("ACF: ", full_file)
  
  
  #-------------------------
  # Zoomed ACF
  #-------------------------
  
  acf_obj <- acf(sigma2_samples, lag.max = lag_max, plot = FALSE)
  
  zoom_file <- file.path(zoom_dir, paste0("acf_sigma2_zoomed.png"))
  png(zoom_file, width = 1600, height = 1200, res = 150)
  
  plot(
    acf_obj$lag[-1],
    acf_obj$acf[-1],
    type = "h",
    ylim = c(-ylim_zoom, ylim_zoom),
    xlab = "Lag",
    ylab = "ACF",
    main = bquote("Zoomed ACF of " ~ sigma^2)
  )
  
  ci <- 1.96 / sqrt(length(sigma2_samples))
  abline(h = c(-ci, ci), col = "blue", lty = 2)
  abline(h = 0, col = "red")
  
  dev.off()
  message("Saved: ", zoom_file)
}


#------------------------------------------------------
# (4.2) Effective Sample Size (ESS)
#------------------------------------------------------
# ESS quantifies the amount of independent information
# contained in autocorrelated MCMC sample.
#------------------------------------------------------


#------------------------------------------------------
# ESS of regression coefficients (beta)
#------------------------------------------------------
# Args:
#   beta_list  : list of MCMC chains, each a matrix (iterations * p)
#   x          : design matrix
#   model_name : character string used to name output
# Returns:
#   Data frame containing ESS for each beta coefficient
#------------------------------------------------------

ess_beta_table <- function(beta_list, x, model_name) {
  
  if (!requireNamespace("coda", quietly = TRUE)) {
    stop("Package 'coda' not installed. Run: install.packages('coda')")
  }
  
  output_dir <- file.path("..", "outputs", model_name, "tables", "ESS")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # convert each chain matrix to mcmc, then to mcmc.list
  mlist <- coda::mcmc.list(lapply(beta_list, coda::mcmc))
  
  ess_vals <- coda::effectiveSize(mlist)  # named vector by column
  Ntot <- nrow(beta_list[[1]]) * length(beta_list)
  ess_vals <- pmin(as.numeric(ess_vals), Ntot)
  
  
  ess_table <- data.frame(
    Parameter = colnames(x),
    ESS = round(as.numeric(ess_vals), 1)
  )
  
  output_file <- file.path(output_dir, "ESS_beta.txt")
  write.table(ess_table, file = output_file, row.names = FALSE, quote = FALSE)

  ess_table
}



#------------------------------------------------------
# ESS table for error variance (sigma^2)
#------------------------------------------------------
# Args:
#   sigma2_list : list of MCMC chains, each a numeric vector (iterations)
#   model_name  : character string used to name output
# Returns:
#   Data frame containing ESS for sigma^2
#------------------------------------------------------

ess_sigma2_table <- function(sigma2_list, model_name) {
  
  if (!requireNamespace("coda", quietly = TRUE)) {
    stop("Package 'coda' not installed. Run: install.packages('coda')")
  }
  
  output_dir <- file.path("..", "outputs", model_name, "tables", "ESS")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  mlist <- coda::mcmc.list(lapply(sigma2_list, coda::mcmc))
  ess_val <- as.numeric(coda::effectiveSize(mlist))
  Ntot <- length(sigma2_list[[1]]) * length(sigma2_list)
  ess_val <- min(ess_val, Ntot)
  
  
  ess_table <- data.frame(Parameter = "Sigma2", ESS = round(ess_val, 1))
  
  output_file <- file.path(output_dir, "ESS_sigma2.txt")
  write.table(ess_table, file = output_file, row.names = FALSE, quote = FALSE)
  
  message("ESS (sigma2) saved to: ", output_file)
  ess_table
}



# ------------------------------------------------------
# (4.3) R-hat (Gelman-Rubin) diagnostic
# ------------------------------------------------------

rhat_scalar <- function(chains) {
  m <- length(chains)
  n <- length(chains[[1]])
  if (m < 2) stop("R-hat needs at least 2 chains.")
  
  chain_means <- sapply(chains, mean)
  chain_vars  <- sapply(chains, var)
  
  W <- mean(chain_vars)
  B <- n * var(chain_means)
  
  var_plus <- (n - 1) / n * W + B / n
  sqrt(var_plus / W)
}

rhat_beta_table <- function(beta_list, x, model_name) {
  output_dir <- file.path("..", "outputs", model_name, "tables", "Rhat")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  p <- ncol(beta_list[[1]])
  rhats <- numeric(p)
  
  for (j in 1:p) {
    chains_j <- lapply(beta_list, function(mat) mat[, j])
    rhats[j] <- rhat_scalar(chains_j)
  }
  
  tab <- data.frame(
    Parameter = colnames(x),
    Rhat = round(rhats, 4)
  )
  
  out <- file.path(output_dir, "Rhat_beta.txt")
  write.table(tab, out, row.names = FALSE, quote = FALSE)
  tab
}

rhat_sigma2_table <- function(sigma2_list, model_name) {
  output_dir <- file.path("..", "outputs", model_name, "tables", "Rhat")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  rhat <- rhat_scalar(sigma2_list)
  
  tab <- data.frame(
    Parameter = "Sigma2",
    Rhat = round(rhat, 4)
  )
  
  out <- file.path(output_dir, "Rhat_sigma2.txt")
  write.table(tab, out, row.names = FALSE, quote = FALSE)
  tab
}
