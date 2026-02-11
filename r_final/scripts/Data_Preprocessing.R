library(dplyr)
library(tidyr)
library(ggplot2)

## (1) Data Cleaning

# (1.1) Missing values:
colSums(is.na(expenses)) # no missing value

# (1.2) Data Types:
str(expenses)
sapply(expenses, typeof)

# (1.3) Convert binary variables to 0/1:
expenses_clean <- expenses %>%
  mutate(
    sex = ifelse(sex == "male", 1, 0), # female = 0, reference category
    smoker = ifelse(smoker == "yes", 1, 0) # nonsmoker = 0, reference category
  )

# (1.4) Remove duplicates:
sum(duplicated(expenses_clean)) # one duplicate row

expenses_clean <- expenses_clean %>% # remove duplicate row
  distinct()

# (1.5) Outliers:
num_expenses <- expenses_clean %>%
  select(where(is.numeric))

long_expeses <- num_expenses %>%
  pivot_longer(
    cols = everything(),
    names_to = "feature",
    values_to = "value"
  )

ggplot(long_expeses, aes(y = value)) +
  geom_boxplot(fill = "steelblue") +
  facet_wrap(~ feature, scales = "free_y") +
  theme_minimal() +
  labs(
    title = "Boxplots of All Numeric Features",
    y = "Value",
    x = ""
  )

count_outliers <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)
  Q3 <- quantile(x, 0.75, na.rm = TRUE)
  IQR <- Q3 - Q1
  
  sum(x < (Q1 - 1.5 * IQR) | x > (Q3 + 1.5 * IQR), na.rm = TRUE)
}

outlier_counts <- sapply(num_expenses, count_outliers)
outlier_counts 

# (1.6) Normalize numeric variables:
expenses_clean <- expenses_clean %>%
  mutate(
    across(
      .cols = c(age, bmi, children),
      .fns = ~ as.numeric(scale(.))
    )
  )
summary(expenses_clean) # just for check

str(expenses_clean)

# Distribution of insurance charges

ggplot(df, aes(x = charges)) +
  geom_histogram(bins = 40, fill = "lightblue", color = "black") +
  labs(
    title = "Distribution of insurance charges",
    x = "charges",
    y = "count"
  ) +
  theme_minimal(base_size = 12)

