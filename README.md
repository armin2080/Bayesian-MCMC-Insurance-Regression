# Bayesian-MCMC-Insurance-Regression

A comprehensive statistical analysis project implementing Bayesian MCMC regression using Gibbs sampling to predict medical insurance costs. This project demonstrates Bayesian inference, convergence diagnostics, and comparison with frequentist methods.

## Project Goal

**Objective**: Implement Bayesian regression with MCMC (Markov Chain Monte Carlo) methods for statistical inference on real-world data.

**Specific Tasks**:
- Implement Gibbs sampling for Bayesian linear regression
- Apply the method to a real medical insurance dataset
- Compare posterior inference with frequentist (OLS) results
- Perform comprehensive convergence diagnostics and model validation

**Dataset**: Medical insurance cost data with 1,337 observations and 7 features (age, sex, BMI, children, smoking status, region)

## Project Structure

```
├── data/                      # Shared datasets
│   └── expenses.csv          # Medical insurance dataset
├── python/                    # Python implementation
│   ├── data_downloader.py    # Script to download dataset from Kaggle
│   ├── notebook.ipynb        # Jupyter notebook with analysis
│   ├── requirements.txt      # Python dependencies
│   ├── outputs/              # Generated plots, tables, and results
│   ├── scripts/              # Python analysis modules
│   └── env/                  # Python virtual environment
└── r/                         # R implementation
    ├── scripts/
    │   ├── Data_Preprocessing.R
    │   └── Model_Setup.R
    └── outputs/              # R analysis outputs
        ├── cor_matrix.txt
        └── ols_output.txt
```

## Getting Started

### Python Setup
```bash
cd python
source env/bin/activate
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

### R Setup
```bash
cd r/scripts
# Install required R packages, then:
Rscript Data_Preprocessing.R
Rscript Model_Setup.R
```

## Methodology & Results

### Phase 1: Data Preprocessing
**Steps:**
1. Load medical insurance dataset (1,338 observations)
2. Check for missing values and duplicates
3. Encode categorical variables (sex, smoker, region)
4. Detect and handle outliers using IQR method
5. Standardize numerical features (age, BMI, children, charges)

**Results:**
- ✓ No missing values found
- ✓ 1 duplicate removed → 1,337 final observations
- ✓ 3 outliers detected in charges (retained for analysis)
- ✓ Categorical variables encoded: sex/smoker as binary, region as 3 dummies
- ✓ Features standardized to mean=0, std=1 for numerical stability

**Key Insight**: Smokers have 3.8× higher average charges ($32,050 vs $8,434)

---

### Phase 2: Prior Specification
**Steps:**
1. Define extremely weak priors for regression coefficients
2. Specify prior for error variance
3. Justify prior choices to ensure data-driven inference

**Priors Selected:**
- **Coefficients (β)**: β ~ N(0, B₀⁻¹) where B₀ = 0.0001·I (precision matrix)
  - Mean of 0 assumes no prior directional bias
  - Precision 10⁻⁴ equivalent to variance 10,000·I (extremely weak, nearly flat prior)
  - Allows data to fully dominate the posterior
  
- **Error Variance (σ²)**: σ² ~ Inverse-Gamma(0.01, 0.01)
  - Shape α=0.01: Extremely weak, nearly uniform prior
  - Scale d=0.01: Ensures proper prior while being maximally non-informative

**Justification**: With n=1,337 observations and extremely weak priors (precision=10⁻⁴), the likelihood completely dominates the posterior. This ensures purely data-driven inference, matching the R implementation approach.

---

### Phase 3: Frequentist Baseline (OLS)
**Steps:**
1. Fit Ordinary Least Squares regression
2. Calculate coefficient estimates and standard errors
3. Evaluate model performance (R², RMSE)

**Results:**
```
Model Performance:
  R² Score:  0.7507 (75.07% variance explained)
  RMSE:      0.4993 (on standardized scale)

Top Predictors:
  smoker:    +1.970 (strongest effect)
  age:       +0.298
  bmi:       +0.171
  children:  +0.047
```

**Interpretation**: The OLS model explains 75% of variance in insurance costs. Smoking status is the dominant predictor, followed by age and BMI.

---

### Phase 4: Bayesian MCMC (Gibbs Sampling)
**Steps:**
1. Implement Gibbs sampler with conjugate priors
2. Run 4 parallel chains with 10,000 iterations each
3. Discard 2,000 warmup samples per chain
4. Generate trace plots for visual convergence assessment

**Sampling Configuration:**
- Iterations per chain: 10,000
- Warmup (burn-in): 2,000
- Number of chains: 4
- Total posterior samples: 32,000

**Results:**
```
Gibbs Sampling Complete:
  Chain 1: 8,000 samples (after warmup)
  Chain 2: 8,000 samples (after warmup)
  Chain 3: 8,000 samples (after warmup)
  Chain 4: 8,000 samples (after warmup)
  Total: 32,000 posterior samples
```

**Trace Plots**: All 9 coefficients + σ² show excellent mixing across chains with no drift or poor convergence patterns.

---

### Phase 5: Convergence Diagnostics
**Steps:**
1. Compute Autocorrelation Function (ACF) for all parameters
2. Calculate Effective Sample Size (ESS) for efficiency assessment
3. Verify rapid ACF decay and high ESS values

**Results:**

**Autocorrelation Analysis:**
- All parameters show rapid ACF decay to near-zero within 50 lags
- No significant long-range autocorrelation detected
- Indicates good mixing and independence of samples

**Effective Sample Size (ESS):**
```
Parameter         ESS      Efficiency
-----------------------------------------
Intercept       22,847      71.40%
age             23,690      74.03%
sex             22,934      71.67%
bmi             23,156      72.36%
children        22,645      70.77%
smoker          21,423      66.95%  (lowest, still good)
region_northwest 22,783     71.20%
region_southeast 22,901     71.57%
region_southwest 22,490     70.28%

Mean ESS:       22,653      70.79%
```

**Interpretation**: Average efficiency of 70.79% means 32,000 MCMC samples provide ~22,653 effectively independent samples—excellent performance indicating minimal autocorrelation.

---

### Phase 6: Bayesian vs Frequentist Comparison
**Steps:**
1. Extract posterior mean coefficients from Bayesian analysis
2. Compare with OLS point estimates
3. Quantify agreement (correlation, differences)

**Results:**
```
Coefficient Comparison:
  Mean Absolute Difference:  0.0001
  Maximum Difference:        0.0002
  Correlation:               1.0000

Example (smoker coefficient):
  OLS:      1.9699
  Bayesian: 1.9700
  Difference: 0.0001 (0.003%)
```

**Key Finding**: Near-perfect agreement between methods validates:
1. ✓ Gibbs sampler implementation is correct
2. ✓ Weakly informative priors allow data to dominate
3. ✓ Large sample size (n=1,337) drives posterior toward MLE
4. ✓ Bayesian framework adds uncertainty quantification (credible intervals)

---

### Phase 7: Posterior Inference
**Steps:**
1. Compute posterior summary statistics (mean, median, std, credible intervals)
2. Interpret 95% credible intervals for each coefficient
3. Assess parameter significance based on interval inclusion of zero

**Results:**
```
Posterior Summary (95% Credible Intervals):
==================================================================
Parameter         Mean    Median   Std      95% CI
------------------------------------------------------------------
Intercept       -0.349   -0.348   0.032   [-0.412, -0.286]
age             +0.298   +0.298   0.014   [+0.271, +0.325]  ✓
sex             -0.011   -0.011   0.028   [-0.065, +0.043]
bmi             +0.171   +0.171   0.014   [+0.143, +0.200]  ✓
children        +0.047   +0.047   0.014   [+0.020, +0.074]  ✓
smoker          +1.970   +1.970   0.034   [+1.903, +2.037]  ✓ (strongest)
region_northwest -0.029  -0.029   0.039   [-0.106, +0.048]
region_southeast -0.085  -0.086   0.040   [-0.164, -0.007]  ✓
region_southwest -0.079  -0.080   0.040   [-0.157, -0.001]  ✓
==================================================================
```

**Interpretation**:
- **Significant predictors** (CIs exclude 0): age, BMI, children, smoker, southeast, southwest
- **Smoking** has the largest effect: +1.97 standardized units
- **Regional effects**: Southeast and southwest have slightly lower costs vs. northeast (baseline)
- **Sex** is not significant (CI includes 0)

---

### Phase 8: Posterior Predictive Checks
**Steps:**
1. Generate posterior predictive distribution (32,000 replicated datasets)
2. Compute 95% prediction intervals
3. Assess coverage: proportion of actual values within intervals
4. Calculate Bayesian RMSE

**Results:**
```
Posterior Predictive Assessment:
  Bayesian RMSE:               0.4993 (identical to OLS)
  95% Prediction Coverage:     94.54%
  Expected Coverage:           95.00%
  
  Model Adequacy: ✓ Excellent
  (Coverage within 93-97% indicates proper calibration)
```

**Interpretation**: 
- Model predictions are well-calibrated (94.54% ≈ 95% target)
- Posterior uncertainty properly captures data variability
- Bayesian framework provides full predictive distribution (not just point estimates)

---

## Summary of Key Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² Score** | 0.7507 | Model explains 75% of variance |
| **RMSE** | 0.4993 | Prediction error on standardized scale |
| **Effective Sample Size** | 22,653 avg | 71% efficiency (excellent) |
| **Prediction Coverage** | 94.54% | Nearly perfect calibration |
| **OLS vs Bayesian Correlation** | 1.000 | Methods agree (validates implementation) |
| **Strongest Predictor** | Smoker (+1.97) | Smoking increases costs most |

### Scientific Insights:
1. **Smoking dominates**: 3.8× higher insurance costs for smokers
2. **Age matters**: Each year adds ~0.30 standardized units to cost
3. **BMI effect**: Higher body mass index → higher costs
4. **Regional variation**: Minimal (southeast/southwest slightly lower)
5. **Bayesian advantage**: Provides full uncertainty quantification via credible intervals and predictive distributions

---

## Data Overview

This project uses the Medical Insurance Cost dataset from Kaggle, containing information about insurance charges based on various factors.

### Dataset Summary
- **Source**: Kaggle (harshsingh2209/medical-insurance-payout)
- **Rows**: 1,338 (after removing duplicates)
- **Columns**: 7 original features + encoded variables

### Features
- **age**: Age of the insured (18-64)
- **sex**: Gender (male=1, female=0)
- **bmi**: Body Mass Index (15.96-53.13)
- **children**: Number of children (0-5)
- **smoker**: Smoking status (yes=1, no=0)
- **region**: Geographic region (encoded as dummies: northwest, southeast, southwest)
- **charges**: Medical insurance costs (target variable)

### Summary Statistics

| Feature  | Mean     | Std Dev | Min     | Max     |
|----------|----------|---------|---------|---------|
| age      | 39.21    | 14.05   | 18.00   | 64.00   |
| bmi      | 30.66    | 6.10    | 15.96   | 53.13   |
| children | 1.09     | 1.21    | 0.00    | 5.00    |
| charges  | 13270.42 | 12110.01| 1121.87 | 63770.43|

### Categorical Distributions

| sex      | Count | Percentage |
|----------|-------|------------|
| Female   | 662   | 49.5%      |
| Male     | 676   | 50.5%      |

| smoker   | Count | Percentage |
|----------|-------|------------|
| No       | 1064  | 79.5%      |
| Yes      | 274   | 20.5%      |

| region   | Count | Percentage |
|----------|-------|------------|
| Southwest| 325   | 24.3%      |
| Southeast| 364   | 27.2%      |
| Northwest| 325   | 24.3%      |
| Northeast| 324   | 24.2%      |

### Key Insights
- Average insurance charge: $13,270
- Smokers have significantly higher charges (mean ~$32,050 vs $8,434 for non-smokers)
- BMI shows a positive correlation with charges
- Age also positively correlates with charges
- Males have slightly higher average charges than females

### Data Cleaning Performed
- Checked for missing values (none found)
- Removed 1 duplicate entry
- Encoded categorical variables (sex/smoker to binary, region to dummies)
- Standardized numerical features for modeling
- Outliers detected but retained for Bayesian analysis