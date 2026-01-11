

## (2) Design matrix
# response:
y <- expenses_clean$charges

# design matrix with intercept:
x <- model.matrix(
  ~ age + sex + bmi + children + smoker,
  data = expenses_clean
)
n <- nrow(x)
p <- ncol(x)

# Baseline Frequentist Fit:
ols_fit <- lm(
  charges ~ age + sex + bmi + children + smoker,
  data = expenses_clean
)

sink("ols_output.txt")
summary(ols_fit)
sink()

# correlation matrix:
cor_matrix <- cor(num_expenses)
sink("cor_matrix.txt")
cor_matrix
sink()


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

beta_trace_plot(beta_list)
sigma2_trace_plot(sigma2_list)

beta_summary_stats(beta_list)
sigma2_summary_stats(sigma2_list)

