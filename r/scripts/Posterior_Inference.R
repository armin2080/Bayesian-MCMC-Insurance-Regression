#-----------------------------------------------------
#(5.1) Posterior Predictive Distribution
#-----------------------------------------------------
# This function generates posterior predictive draws
# y_rep ~ p(y_new | beta, sigma^2)
#
# Args:
#   beta_list   : list of posterior draws of regression coefficients (by chain)
#   sigma2_list : list of posterior draws of error variance (by chain)
#   X_new       : design matrix for new or observed covariates
# Return:
#   y_rep : matrix of posterior predictive draws
#             (rows = posterior draws, columns = observations)
#-----------------------------------------------------

posterior_predictive <- function(beta_list, sigma2_list, X_new) {
  
  # combine chains into a single posterior sample
  beta_all <- do.call(rbind, beta_list)
  sigma2_all <- unlist(sigma2_list)
  
  n_draws <- nrow(beta_all)
  n_new <- nrow(X_new)
  
  # matrix to store replicated outcomes
  y_rep <- matrix(NA, n_draws, n_new)
  
  for ( i in 1:n_draws) {
    # linear predictor
    mu <- X_new %*% beta_all[i, ]
    
    # posterior predictive draw
    y_rep[i, ] <- rnorm(n_new, mu, sqrt(sigma2_all[i]))
  }
  
  return(y_rep)
}

#---------------------------------------------
#(5.2) Posterior predictive checks
#---------------------------------------------
# This function compares observed responses with
# posterior predictive means to assess model fit.
#
# Args:
#   y_obs      : observed response vector
#   y_rep      : posterior predictive matrix from posterior_predictive()
#   model_name : character string used to name output
# Output:
#   ggplot object for visualization
#---------------------------------------------

ppc_plot <- function(y_obs, y_rep, model_name) {
  
  plot_dir <- file.path("..", "outputs", model_name, "plots", "PPC")
  if (!dir.exists(plot_dir))
    dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)
  
  df <- data.frame(
    observed = y_obs,
    predicted = colMeans(y_rep)
  )
  
  file_name <- file.path(plot_dir, "PPC_Scatter.png")
  png(filename = file_name, width = 1600, height = 1200, res = 150)
  
  print(
    ggplot(df, aes(x = observed, y = predicted)) +
    geom_point(alpha = 0.4) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    theme_minimal() +
    labs(
      title = "Posterior Predictive Check",
      x = "Observed y",
      y = "Posterior mean prediction"
    )
  )
  
  dev.off()
  message("saved: ", file_name)
}


#----------------------------------------
# (2.3) Density Overlay PPC
#----------------------------------------

PPC_density_overlay <- function(y_obs, y_rep, model_name) {
  
  plot_dir <- file.path("..", "outputs", model_name, "plots", "PPC")
  if (!dir.exists(plot_dir))
    dir.create(plot_dir, recursive = TRUE)
  
  y_rep_sample <- as.vector(y_rep[sample(nrow(y_rep), 200), ])
  
  df <- data.frame(
    value = c(y_obs, y_rep_sample),
    type = c(
      rep("Observed", length(y_obs)),
      rep("Replicated", length(y_rep_sample))
    )
  )
  
  file_name <- file.path(plot_dir, "PPC_Density_Overlay.png")
  png(file_name, width = 1600, height = 1200, res =150)
  
  print(
    ggplot(df, aes(x = value, fill = type)) +
      geom_density(alpha = 0.4) +
      theme_minimal() +
      labs(
        title = "Posterior Predictive Density Overlay",
        x = "y",
        y = "Density"
      )
  )
  dev.off()
  message("Saved: ", file_name)
}


#------------------------------------------
# (2.4) Residuals Posterior Predictive Check
#------------------------------------------

ppc_residual_plot <- function(y_obs, y_rep, model_name) {
  
  plot_dir <- file.path("..", "outputs", model_name, "plots", "PPC")
  if (!dir.exists(plot_dir))
    dir.create(plot_dir, recursive = TRUE)
  
  predict_mean <- colMeans(y_rep)
  residuals <- y_obs - predict_mean
  
  df <- data.frame(
    predicted = predict_mean,
    residual = residuals
  )
  
  file_name <- file.path(plot_dir, "PPC_Residuals.png")
  png(file_name, width = 1600, height = 1200, res = 150)
  
  print(
    ggplot(df, aes(x = predicted, y = residual)) +
      geom_point(alpha = 0.4) +
      theme_minimal() +
      labs(
        title = "Posterior Predictive Residual Check",
        x = "Posterior Mean Prediction",
        y = "Predictive Residual"
      )
  )
  
  dev.off()
  message("Saved: ", file_name)
}
