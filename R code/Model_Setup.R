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

