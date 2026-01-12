#-----------------------------------------
#(4.1) Autocorrelation Plots
#-----------------------------------------

          #-------ACF Plot for beta--------

acf_plot_beta <- function(beta_list, model_name, lag_max = 50) {
  plot_dir <- file.path("plots", "ACF_plots")
  if (!dir.exists(plot_dir))
    dir.create(plot_dir, recursive = TRUE)
  
  p <- ncol(beta_list[[1]])
  
  for (j in 1:p) {
    filename <- file.path(plot_dir, paste0("afc_beta_", j-1, ".png"))
    png(filename, width = 1600, height = 1200, res = 150)
    
    acf(
      do.call(c, lapply(beta_list, function(x) x[, j])),
      lag.max = lag_max,
      main = bquote("ACF of " ~ beta[.(j-1)])
    )
    
    dev.off()
    message("Saved: ", filename)
  }
}

          #--------ACF Plot for sigma2--------

acf_plot_sigma2 <- function(sigma2_list, model_name, lag_max = 50) {
  
  plot_dir <- file.path("plots", "ACF_plots")
  if (!dir.exists(plot_dir))
    dir.create(plot_dir, recursive = TRUE)
  
  filename <- file.path(plot_dir, "acf_sigma2.png")
  png(filename, width = 1600, height = 1200, res = 150)
  
  acf(
    unlist(sigma2_list),
    lag.max = lag_max,
    main = expression("ACF of " ~ sigma^2)
  )
  dev.off()
  message("ACF: ", filename)
}

#-------------------------------------
#(4.2) Effective Sample Size(ESS)
#-------------------------------------

effective_sample_size <- function(x, max_lag = 100) {
  acf_vals <- acf(x, lag.max = max_lag, plot = FALSE)$acf[-1]
  
  # Initial positive sequence
  positive_acf <- acf_vals[acf_vals > 0]
  
  ess <- length(x) / (1 + 2 * sum(positive_acf))
  return(ess)
}

        #----------ESS of beta--------------

ess_beta_table <- function(beta_list, x, model_name) {
  
  output_dir <- file.path("r", "outputs", model_name, "ESS_tables")
  if (!dir.exists(output_dir))
    dir.create(output_dir, recursive = TRUE)
  
  beta_all <- do.call(rbind, beta_list)
  
  ess_vals <- apply(beta_all, 2, effective_sample_size)
  
  ess_table <- data.frame(
    Parameter = colnames(x),
    ESS = round(ess_vals, 1)
  )
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

        #----------ESS of sigma2------------

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
