# (2.1) Design matrix
# response:
y <- expenses_clean$charges

# design matrix with intercept:
x <- model.matrix(
  ~ age + sex + bmi + children + smoker,
  data = expenses_clean
)

n <- nrow(x)
p <- ncol(x)

# (2.2) Baseline Frequentist Fit:
ols_fit <- lm(
  charges ~ age + sex + bmi + children + smoker,
  data = expenses_clean 
) # the effect of "sex" is non-significant

sink("ols_output.txt")
summary(ols_fit)
sink()

#(2.3) correlation matrix:
cor_matrix <- cor(num_expenses)
sink("cor_matrix.txt")
cor_matrix
sink()
# "sex" has a very small correlation with "charges"

#------------------------------------
# Run Gibbs Sampler & Trace Plot
#------------------------------------

source("r/scripts/Gibbs_Sampling.R")

# Run Gibbs sampler
set.seed(123)
gibbs_result <- gibbs_lm(
  y = y,        
  X = x,        
  n_iter = 10000, 
  warmup = 2000 ,
  n_chains = 4
)

beta_list <- lapply(gibbs_result, function(chain) chain$beta)
sigma2_list <- lapply(gibbs_result, function(chain) chain$sigma2)

# Saves trace plot images for each model in its own folder to avoid file overwriting. Pass the model name as an argument
beta_trace_plot(beta_list,"gibbs_result")
sigma2_trace_plot(sigma2_list,"gibbs_result")

beta_summary_stats(beta_list)
sigma2_summary_stats(sigma2_list)

#-------------------------------------------
# Run Autocorrelation plots
#-------------------------------------------

# Load Convergence function
source("r/scripts/Convergence_Detection.R")

# ACF plots
acf_plot_beta(beta_list,"gibbs_result")
acf_plot_sigma2(sigma2_list, "gibbs_result")

# ESS
ess_beta_table(beta_list, x, "gibbs_result")
ess_sigma2_table(sigma2_list, "gibbs_result")

# R-hat
rhat_beta_table(beta_list, x, "gibbs_result")
rhat_sigma2_table(sigma2_list, "gibbs_result")



#----------------------------------------------
# Run Posterior Predictive
#----------------------------------------------

# Load PPC function
source("r/scripts/Posterior_Inference.R")

# PPC
# use training design matrix for PPC : X_new = x
y_rep <- posterior_predictive(
  beta_list = beta_list,
  sigma2_list = sigma2_list,
  X_new = x)

# PPC plot
ppc_plot(y, y_rep, "gibbs_result")
PPC_density_overlay(y, y_rep, "gibbs_result")
ppc_residual_plot(y, y_rep, "gibbs_result")

#===============================================
# log Transformation
#===============================================

# response:
y_log <- log(expenses_clean$charges)

# design matrix with intercept:
x <- model.matrix(
  ~ age + sex + bmi + children + smoker,
  data = expenses_clean
)

n <- nrow(x)
p <- ncol(x)

#------------------------------------
# Run Gibbs Sampler & Trace Plot
#------------------------------------

source("r/scripts/Gibbs_Sampling.R")

# Run Gibbs sampler
set.seed(123)
log_fit <- gibbs_lm(
  y = y_log,        
  X = x,        
  n_iter = 10000, 
  warmup = 2000 ,
  n_chains = 4
)

beta_list <- lapply(log_fit, function(chain) chain$beta)
sigma2_list <- lapply(log_fit, function(chain) chain$sigma2)

# Saves trace plot images for each model in its own folder to avoid file overwriting. Pass the model name as an argument
beta_trace_plot(beta_list,"log_fit")
sigma2_trace_plot(sigma2_list,"log_fit")

beta_summary_stats(beta_list)
sigma2_summary_stats(sigma2_list)

#-------------------------------------------
# Run Autocorrelation plots
#-------------------------------------------

# Load Convergence function
source("r/scripts/Convergence_Detection.R")

# ACF plots
acf_plot_beta(beta_list,"log_fit")
acf_plot_sigma2(sigma2_list, "log_fit")

# ESS
ess_beta_table(beta_list, x, "log_fit")
ess_sigma2_table(sigma2_list, "log_fit")

# R-hat
rhat_beta_table(beta_list, x, "log_fit")
rhat_sigma2_table(sigma2_list, "log_fit")



#----------------------------------------------
# Run Posterior Predictive
#----------------------------------------------

# Load PPC function
source("r/scripts/Posterior_Inference.R")

# PPC
# use training design matrix for PPC : X_new = x
y_rep <- posterior_predictive(
  beta_list = beta_list,
  sigma2_list = sigma2_list,
  X_new = x)

# PPC plot
ppc_plot(y_log, y_rep, "log_fit")
PPC_density_overlay(y_log, y_rep, "log_fit")
ppc_residual_plot(y_log, y_rep, "log_fit")


#==================================================
# Interaction Model
#==================================================

# response:
y_log <- log(expenses_clean$charges)

# design matrix with interaction:
x_int <- model.matrix(
  ~ age + sex + bmi + children + smoker + smoker:bmi,
  data = expenses_clean
)

n <- nrow(x_int)
p <- ncol(x_int)


#------------------------------------
# Run Gibbs Sampler & Trace Plot
#------------------------------------

source("r/scripts/Gibbs_Sampling.R")

# Run Gibbs sampler
set.seed(123)
interaction <- gibbs_lm(
  y = y_log,        
  X = x_int,        
  n_iter = 10000, 
  warmup = 2000 ,
  n_chains = 4
)

beta_list <- lapply(interaction, function(chain) chain$beta)
sigma2_list <- lapply(interaction, function(chain) chain$sigma2)

# Saves trace plot images for each model in its own folder to avoid file overwriting. Pass the model name as an argument
beta_trace_plot(beta_list,"interaction")
sigma2_trace_plot(sigma2_list,"interaction")

beta_summary_stats(beta_list)
sigma2_summary_stats(sigma2_list)

#-------------------------------------------
# Run Autocorrelation plots
#-------------------------------------------

# Load Convergence function
source("r/scripts/Convergence_Detection.R")

# ACF plots
acf_plot_beta(beta_list,"interaction")
acf_plot_sigma2(sigma2_list, "interaction")

# ESS
ess_beta_table(beta_list, x_int, "interaction")
ess_sigma2_table(sigma2_list, "interaction")

# R-hat
rhat_beta_table(beta_list, x_int, "interaction")
rhat_sigma2_table(sigma2_list, "interaction")



#----------------------------------------------
# Run Posterior Predictive
#----------------------------------------------

# Load PPC function
source("r/scripts/Posterior_Inference.R")

# PPC
# use training design matrix for PPC : X_new = x
y_rep <- posterior_predictive(
  beta_list = beta_list,
  sigma2_list = sigma2_list,
  X_new = x_int)

# PPC plot
ppc_plot(y_log, y_rep, "interaction")
PPC_density_overlay(y_log, y_rep, "interaction")
ppc_residual_plot(y_log, y_rep, "interaction")



# ==================================================
# Prior Sensitivity (run on best model: interaction)
# ==================================================

p <- ncol(x_int)

priors <- list(
  diffuse = list(B0 = diag(1e-4, p), a0 = 0.01, d0 = 0.01),
  mild    = list(B0 = diag(1e-2, p), a0 = 2,    d0 = 1),
  strong  = list(B0 = diag(1e-1, p), a0 = 2,    d0 = 1)
)

prior_summaries <- lapply(names(priors), function(pr_name) {
  pr <- priors[[pr_name]]
  fit <- gibbs_lm(
    y = y_log,
    X = x_int,
    n_iter = 10000,
    warmup = 2000,
    n_chains = 4,
    B0 = pr$B0,
    a0 = pr$a0,
    d0 = pr$d0
  )
  beta_list_pr <- lapply(fit, function(ch) ch$beta)
  
  summ <- beta_summary_stats(beta_list_pr)
  out <- data.frame(
    Prior = pr_name,
    Parameter = colnames(x_int),
    Mean = summ[, "Mean"],
    SD = summ[, "SD"],
    Q2.5 = summ[, "2.5%"],
    Q97.5 = summ[, "97.5%"]
  )
  out
})

prior_summaries <- do.call(rbind, prior_summaries)

outdir <- file.path("r", "outputs", "prior_sensitivity")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)
write.csv(prior_summaries, file.path(outdir, "prior_sensitivity_beta.csv"), row.names = FALSE)

