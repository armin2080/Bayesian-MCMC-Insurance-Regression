# R Implementation: Bayesian MCMC Insurance Regression

This directory contains the complete R implementation of the Bayesian MCMC analysis, mirroring the Python implementation.

## 📁 Directory Structure

```
r/
├── analysis.Rmd              # R Markdown notebook (equivalent to notebook.ipynb)
├── outputs/                  # Analysis results and plots
└── scripts/                  # R analysis modules
    ├── Algorithm_Comparison.R       # Compare Gibbs vs MH
    ├── Convergence_Detection.R      # R-hat, ESS diagnostics
    ├── Data_Preprocessing.R         # Data loading and cleaning
    ├── Gibbs_Sampling.R            # Gibbs sampler implementation
    ├── Metropolis_Hastings.R       # MH sampler with adaptive tuning
    ├── Model_Setup.R               # Prior specification and model setup
    ├── Posterior_Inference.R       # Posterior estimates and credible intervals
    ├── Quick_Analysis.R            # Fast baseline analysis
    ├── Residual_Diagnostics.R      # Model diagnostics and residual plots
    ├── Run_Full_Analysis.R         # Master script for complete pipeline
    ├── Test_Installation.R         # Environment verification
    └── data_downloader.R           # Kaggle dataset downloader
```

## 🚀 Quick Start

### 1. Test Installation

Verify that all required R packages are installed:

```r
source("scripts/Test_Installation.R")
```

This checks for:
- R version (>= 3.6)
- Required packages: `MASS`, `coda`, `ggplot2`, `gridExtra`, `lmtest`
- Data files availability
- Script files integrity

### 2. Quick Analysis

Run a fast Gibbs sampling analysis:

```r
source("scripts/Quick_Analysis.R")
```

**Output:**
- Posterior estimates
- Convergence diagnostics (R-hat, ESS)
- Model fit statistics (R², RMSE)
- Results saved to `outputs/quick_analysis/`

### 3. Full Analysis Pipeline

Run the complete analysis with both algorithms:

```r
source("scripts/Run_Full_Analysis.R")
```

**Includes:**
1. Data preprocessing
2. Gibbs sampling (3 chains, 50k iterations)
3. Metropolis-Hastings (3 chains, 50k iterations)
4. Algorithm comparison
5. Residual diagnostics
6. Comprehensive reporting

### 4. Interactive Analysis (R Markdown)

Open and knit the R Markdown notebook:

```r
# In RStudio:
rmarkdown::render("analysis.Rmd")

# Or use knitr:
library(knitr)
knit("analysis.Rmd")
```

**Output:** HTML report with interactive visualizations

## 📦 Required Packages

Install all required packages:

```r
install.packages(c("MASS", "coda", "ggplot2", "gridExtra", "lmtest", "knitr", "rmarkdown"))
```

### Package Descriptions

| Package | Purpose |
|---------|---------|
| `MASS` | Multivariate normal sampling (`mvrnorm`) |
| `coda` | MCMC diagnostics (ESS, R-hat, Gelman-Rubin) |
| `ggplot2` | Advanced visualizations |
| `gridExtra` | Multi-panel plot layouts |
| `lmtest` | Residual diagnostic tests (optional) |
| `knitr` | R Markdown rendering |
| `rmarkdown` | Document generation |

## 🔬 Analysis Modules

### Core MCMC Samplers

#### Gibbs Sampling (`Gibbs_Sampling.R`)

```r
results <- gibbs_lm(
  y, X,
  prior_beta_mean, prior_beta_precision,
  prior_sigma2_shape, prior_sigma2_scale,
  n_iter = 50000, warmup = 10000,
  seed = 42
)
```

**Features:**
- Conjugate prior sampling (Normal-Inverse-Gamma)
- Block updates for β and σ²
- ~21× more efficient than MH (ESS per second)

#### Metropolis-Hastings (`Metropolis_Hastings.R`)

```r
results <- metropolis_hastings_lm(
  y, X,
  prior_beta_mean, prior_beta_precision,
  prior_sigma2_shape, prior_sigma2_scale,
  n_iter = 50000, warmup = 10000,
  seed = 42
)
```

**Features:**
- Adaptive proposal tuning (Roberts & Rosenthal optimal rates)
- Log-transform for σ² with Jacobian adjustment
- Target acceptance rates: β=23.4%, σ²=44%

### Diagnostics and Comparison

#### Convergence Detection (`Convergence_Detection.R`)

- **Gelman-Rubin R-hat**: Multi-chain convergence diagnostic
- **Effective Sample Size (ESS)**: Independent samples estimate
- Target: R-hat < 1.1, ESS > 400

#### Algorithm Comparison (`Algorithm_Comparison.R`)

```r
compare_algorithms(gibbs_results, mh_results, param_names, output_dir)
```

**Outputs:**
- ESS comparison table
- Computational efficiency (ESS/second)
- Posterior agreement
- Acceptance rates (MH only)
- Multi-panel comparison plots

#### Residual Diagnostics (`Residual_Diagnostics.R`)

```r
plot_residual_diagnostics(y, X, beta_samples, sigma2_samples, output_path)
```

**Diagnostic Plots:**
1. Residuals vs Fitted (homoscedasticity check)
2. Normal Q-Q Plot (normality assumption)
3. Scale-Location (variance stability)
4. Residual Histogram (distribution shape)

**Statistical Tests:**
- Shapiro-Wilk (normality)
- Durbin-Watson (autocorrelation)
- Breusch-Pagan (heteroscedasticity)

### Data and Setup

#### Data Preprocessing (`Data_Preprocessing.R`)

Pipeline:
1. Load raw data (`expenses.csv`)
2. Encode binary variables (sex, smoker)
3. Remove duplicates
4. Standardize numeric features (optional)
5. Drop region variable (avoid hierarchical complexity)

#### Model Setup (`Model_Setup.R`)

Prior specifications:
- **β ~ Normal(0, 1000·I)**: Weakly informative
- **σ² ~ Inverse-Gamma(0.01, 0.01)**: Diffuse prior
- Design matrix: `X = [1, age, sex, bmi, children, smoker]`

### Posterior Inference (`Posterior_Inference.R`)

Functions:
- `posterior_summary()`: Mean, SD, 95% credible intervals
- `posterior_predictive()`: Predictive distribution
- `hpd_interval()`: Highest posterior density intervals

## 📊 Output Structure

```
outputs/
├── quick_analysis/
│   ├── posterior_estimates.csv
│   └── model_fit.csv
├── gibbs_result/
│   ├── ACF_plots/              # Autocorrelation plots
│   ├── PPC/                    # Posterior predictive checks
│   ├── ESS_tables/             # Effective sample sizes
│   ├── Rhat_tables/            # Convergence diagnostics
│   └── residual_diagnostics.png
├── mh_baseline/
│   └── (same structure as gibbs_result)
├── algorithm_comparison/
│   ├── ess_comparison.csv
│   ├── efficiency_comparison.csv
│   ├── rhat_comparison.csv
│   ├── posterior_estimates_comparison.csv
│   ├── mh_acceptance_rates.csv
│   └── algorithm_comparison_summary.png
└── final_summary/
    ├── posterior_summary.csv
    └── model_fit.csv
```

## 🎯 Key Results

### Posterior Estimates (Gibbs Sampling, 50k iterations, 3 chains)

| Parameter | Mean | 95% CI |
|-----------|------|--------|
| Intercept | $8,461 | [$7,392, $9,531] |
| Age | $3,619 | [$3,489, $3,749] |
| Sex | -$131 | [-$652, $390] |
| BMI | $1,966 | [$1,838, $2,094] |
| Children | $572 | [$331, $813] |
| **Smoker** | **$23,823** | [**$23,268**, **$24,378**] |

### Model Performance

- **R² = 0.7496** (75% variance explained)
- **RMSE = $6,058**
- **MAE = $4,120**
- **All chains converged** (R-hat = 1.000 for all parameters)

### Algorithm Efficiency

| Algorithm | Avg ESS | Time (s) | ESS/sec | Efficiency Ratio |
|-----------|---------|----------|---------|------------------|
| Gibbs | 35,000 | 18.5 | 1,892 | **21.1×** |
| MH | 3,100 | 34.7 | 89 | 1.0× |

**Conclusion:** Gibbs sampling is ~21× more computationally efficient for this conjugate prior model.

## 🔍 Comparison with Python Implementation

### Structural Equivalence

| Python | R | Purpose |
|--------|---|---------|
| `notebook.ipynb` | `analysis.Rmd` | Interactive analysis notebook |
| `data_preprocessing.py` | `Data_Preprocessing.R` | Data cleaning pipeline |
| `gibbs_sampling.py` | `Gibbs_Sampling.R` | Gibbs sampler |
| `metropolis_hastings.py` | `Metropolis_Hastings.R` | MH sampler |
| `algorithm_comparison.py` | `Algorithm_Comparison.R` | MCMC comparison |
| `convergence_detection.py` | `Convergence_Detection.R` | Diagnostics |
| `residual_diagnostics.py` | `Residual_Diagnostics.R` | Residual analysis |
| `posterior_inference.py` | `Posterior_Inference.R` | Inference tools |

### Implementation Differences

1. **Libraries:**
   - Python: NumPy, SciPy, Pandas, Matplotlib
   - R: MASS, coda, ggplot2, base R

2. **MCMC Diagnostics:**
   - Python: Custom ESS/R-hat implementations
   - R: `coda` package (industry standard)

3. **Visualization:**
   - Python: Matplotlib, seaborn
   - R: ggplot2 (grammar of graphics)

4. **Notebook Format:**
   - Python: Jupyter (.ipynb)
   - R: R Markdown (.Rmd) → HTML/PDF

### Validation

Both implementations produce **identical results** (within Monte Carlo error):

- Posterior means agree to 4 decimals
- R-hat = 1.000 in both
- RMSE difference < $1
- ESS ratio: ~21× in both

## 📖 Usage Examples

### Example 1: Custom Prior Analysis

```r
source("scripts/Gibbs_Sampling.R")

# Load data
df <- read.csv("../data/expenses_cleaned.csv")
y <- df$charges
X <- cbind(1, as.matrix(df[, c("age", "sex", "bmi", "children", "smoker")]))

# Set informative priors
prior_beta_mean <- c(10000, 250, 0, 300, 500, 20000)  # Domain knowledge
prior_beta_precision <- diag(c(0.01, 0.1, 0.1, 0.1, 0.1, 0.01))
prior_sigma2_shape <- 3
prior_sigma2_scale <- 100000

# Run Gibbs sampler
results <- gibbs_lm(y, X, prior_beta_mean, prior_beta_precision,
                    prior_sigma2_shape, prior_sigma2_scale,
                    n_iter = 100000, warmup = 20000, seed = 123)

# Extract posterior
beta_posterior <- results$beta
sigma2_posterior <- results$sigma2
```

### Example 2: Posterior Predictive Sampling

```r
# New individual
x_new <- c(1, 35, 1, 28, 1, 1)  # 35 yo male, BMI=28, 1 child, smoker

# Sample from posterior predictive
n_samples <- 5000
y_pred_samples <- numeric(n_samples)

for (i in 1:n_samples) {
  beta_i <- beta_posterior[i, ]
  sigma2_i <- sigma2_posterior[i]
  
  # Predictive mean
  mu_i <- sum(x_new * beta_i)
  
  # Sample from predictive distribution
  y_pred_samples[i] <- rnorm(1, mu_i, sqrt(sigma2_i))
}

# 95% prediction interval
pred_interval <- quantile(y_pred_samples, c(0.025, 0.975))
cat(sprintf("Predicted charges: $%.2f [95%% PI: $%.2f - $%.2f]\n",
            mean(y_pred_samples), pred_interval[1], pred_interval[2]))
```

### Example 3: Parallel Chain Execution

```r
library(parallel)

# Detect cores
n_cores <- detectCores() - 1
n_chains <- n_cores

# Run chains in parallel
results <- mclapply(1:n_chains, function(chain_id) {
  gibbs_lm(y, X, prior_beta_mean, prior_beta_precision,
           prior_sigma2_shape, prior_sigma2_scale,
           n_iter = 50000, warmup = 10000,
           seed = 42 + chain_id)
}, mc.cores = n_cores)
```

## 🛠️ Troubleshooting

### Common Issues

**1. Package installation errors:**
```r
# Try different CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org"))
install.packages("package_name")
```

**2. Memory issues with large MCMC samples:**
```r
# Use thinning
results <- gibbs_lm(..., n_iter = 100000, thin = 10)  # Keep every 10th sample
```

**3. Convergence problems:**
```r
# Increase warmup period
results <- gibbs_lm(..., n_iter = 100000, warmup = 30000)

# Or run longer chains
results <- gibbs_lm(..., n_iter = 200000, warmup = 50000)
```

**4. Slow execution:**
```r
# Use fewer iterations for testing
results <- gibbs_lm(..., n_iter = 10000, warmup = 2000)

# Or parallelize chains (see Example 3)
```

## 📚 References

### MCMC Theory
- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapman & Hall.
- Brooks, S., et al. (2011). *Handbook of Markov Chain Monte Carlo*. CRC Press.

### R Packages
- Plummer, M., et al. (2006). CODA: Convergence Diagnosis and Output Analysis for MCMC. *R News*, 6(1), 7-11.
- Venables, W. N., & Ripley, B. D. (2002). *Modern Applied Statistics with S* (4th ed.). Springer.

### Convergence Diagnostics
- Gelman, A., & Rubin, D. B. (1992). Inference from iterative simulation using multiple sequences. *Statistical Science*, 7(4), 457-472.
- Roberts, G. O., & Rosenthal, J. S. (2001). Optimal scaling for various Metropolis-Hastings algorithms. *Statistical Science*, 16(4), 351-367.

## 📝 License

This project is part of a Bayesian Statistics course. See [LICENSE](../LICENSE) for details.

## 👥 Contributors

- R implementation: Translation of Python codebase
- Original Python analysis: Bayesian MCMC project team

## 🔗 Related Files

- **Main README**: [../README.md](../README.md)
- **Python implementation**: [../python/](../python/)
- **Final report**: [../Final_Report/Report.pdf](../Final_Report/Report.pdf)
- **Dataset**: [../data/expenses_cleaned.csv](../data/expenses_cleaned.csv)
