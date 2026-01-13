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
  plot_dir <- file.path("plots",model_name, "ACF_plots")
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
  
  plot_dir <- file.path("plots",model_name, "ACF_plots")
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
# core ESS computation using the initial positive sequence
#------------------------------------------------------
# Args:
#   x       : numeric vector of MCMC samples
#   max_lag : maximum lag for autocorrelation
# Returns:
#   effective sample size (ESS)
#------------------------------------------------------

effective_sample_size <- function(x, max_lag = 100) {
  
  # compute autocorrelation values
  acf_vals <- acf(x, lag.max = max_lag, plot = FALSE)$acf[-1]
  
  # Initial positive sequence 
  positive_acf <- acf_vals[acf_vals > 0]
  
  ess <- length(x) / (1 + 2 * sum(positive_acf))
  return(ess)
}


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
  
  output_dir <- file.path("r", "outputs", model_name, "ESS_tables")
  if (!dir.exists(output_dir))
    dir.create(output_dir, recursive = TRUE)
  
  # combine all chains into one matrix
  beta_all <- do.call(rbind, beta_list)
  
  # compute ESS for each coefficient
  ess_vals <- apply(beta_all, 2, effective_sample_size)
  
  ess_table <- data.frame(
    Parameter = colnames(x),
    ESS = round(ess_vals, 1)
  )
  
  # save ESS table
  output_file <- file.path(output_dir, "ESS_beta.txt")
  write.table(
    ess_table,
    file = output_file,
    row.names = FALSE,
    col.names = TRUE,
    quote = FALSE
  )
  
  message("ESS (beta) saved to: ", output_file)
  return(ess_table)
}


#------------------------------------------------------
# ESS table for error variance (sigma^2)
#------------------------------------------------------
# Args:
#   sigma2_list : list of MCMC chains, each a matrix (iterations * p)
#   model_name  : character string used to name output
# Returns:
#   Data frame containing ESS for sigma^2
#------------------------------------------------------

ess_sigma2_table <- function(sigma2_list, model_name) {
  output_dir <- file.path("r", "outputs", model_name, "ESS_tables")
  if (!dir.exists(output_dir))
    dir.create(output_dir, recursive = TRUE)
  
  ess_vals <-round(effective_sample_size(unlist(sigma2_list)), 1)
  
  ess_table <- data.frame(
    Parameter = "Sigma2",
    ESS = ess_vals
  )
  
  output_file <- file.path(output_dir, "ESS_sigma2.txt")
  write.table(
    ess_table,
    file = output_file,
    row.names = FALSE,
    col.names = TRUE,
    quote = FALSE
  )
  
  message("ESS (sigma^2) saved to: ", output_file)
  return(ess_table)
}
