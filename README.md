# MCMC Algorithm Comparison: Gibbs Sampling vs Metropolis-Hastings

**Course:** On the Theory and Practice of Monte Carlo Simulations  
**Focus:** Algorithm Design, Implementation, and Comparison (Not Bayesian Modeling)

## Project Goal

**Primary Objective**: Implement and compare two MCMC algorithms from scratch to demonstrate deep understanding of Monte Carlo simulation methods.

**Key Focus Areas**:
1. **Algorithm Design**: Implement Gibbs Sampling and Metropolis-Hastings manually
2. **Mathematical Derivations**: Derive conditional distributions and proposal distributions
3. **Performance Comparison**: Compare convergence, efficiency, and computational cost
4. **Implementation Quality**: Show understanding beyond black-box tools

**Application**: Bayesian linear regression on medical insurance data (1,337 observations)

**Emphasis**: This is a **Monte Carlo Simulation course**, not a Bayesian Modeling course. The focus is on understanding how MCMC algorithms work, not on achieving perfect model fit.

## Project Structure

```
├── ALGORITHM_DERIVATIONS.md  # Complete mathematical derivations (CRITICAL)
├── data/                      # Shared datasets
│   └── expenses.csv          # Medical insurance dataset
├── python/                    # Python implementation
│   ├── scripts/              # Core algorithm implementations
│   │   ├── gibbs_sampling.py           # Gibbs sampler (manual implementation)
│   │   ├── metropolis_hastings.py      # MH sampler (manual implementation)
│   │   ├── algorithm_comparison.py     # Algorithm comparison utilities
│   │   ├── data_preprocessing.py       # Data handling
│   │   └── convergence_detection.py    # Diagnostics
│   ├── notebook.ipynb        # Analysis notebook
│   ├── requirements.txt      # Python dependencies
│   └── outputs/              # Generated plots and results
│       ├── gibbs_result/     # Gibbs diagnostics
│       ├── mh_result/        # MH diagnostics
│       └── algorithm_comparison/  # Comparative analysis
└── Final_Report/
    └── Report.tex            # LaTeX report (algorithm-focused)
```

## Key Documents

1. **`ALGORITHM_DERIVATIONS.md`**: Complete mathematical derivations showing:
   - Full conditional distributions for Gibbs sampling
   - Acceptance ratios for Metropolis-Hastings
   - Proposal distribution design
   - Jacobian corrections
   - Implementation details

2. **Implementation files**: Manual implementations without relying on black-box MCMC libraries

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

## Algorithms Implemented

This project implements **two MCMC algorithms from scratch**:

### 1. **Gibbs Sampling**
- Exploits conjugate Normal-Inverse-Gamma priors
- Samples from full conditional distributions
- 100% acceptance rate
- Requires mathematical derivation of conditionals
- **See:** `python/scripts/gibbs_sampling.py` and detailed derivations in `ALGORITHM_DERIVATIONS.md`

### 2. **Metropolis-Hastings**
- General-purpose MCMC (no conjugacy required)
- Random Walk proposals with adaptive tuning
- Requires proposal distribution design
- Includes Jacobian correction for log-transformed variance
- **See:** `python/scripts/metropolis_hastings.py` and detailed derivations in `ALGORITHM_DERIVATIONS.md`

### Algorithm Comparison
Comprehensive comparison on:
- Effective Sample Size (ESS)
- Convergence speed (R-hat)
- Computational efficiency (ESS per second)
- Acceptance rates
- Autocorrelation structure

**See:** `python/scripts/algorithm_comparison.py` for implementation

## Methodology & Results

### Phase 1: Mathematical Foundations

**Objective**: Derive MCMC algorithms from first principles

**Gibbs Sampling Derivations:**
1. Derive full conditional for β: Complete the square to get p(β|σ²,y) ~ N(bₙ, σ²Bₙ⁻¹)
2. Derive full conditional for σ²: Combine likelihood and priors to get p(σ²|β,y) ~ IG(aₙ, dₙ)
3. Prove that sampling from these conditionals preserves the joint posterior

**Metropolis-Hastings Derivations:**
1. Design proposal distribution for β: Random walk q(β*|β) ~ N(β, τ²σ²I)
2. Design proposal for σ²: Log-normal proposal to ensure positivity
3. Derive acceptance ratio with Jacobian correction
4. Prove detailed balance condition

**Key Achievement**: Complete manual derivation without relying on textbook formulas blindly

---

### Phase 2: Algorithm Implementation

**Objective**: Implement both algorithms from scratch (no black-box MCMC libraries)

**Implementation Details:**

**Gibbs Sampler:**
- Direct sampling from full conditionals
- Cholesky decomposition for numerical stability
- Matrix operations optimized for repeated use

**Metropolis-Hastings Sampler:**
- Random walk proposals
- Log-scale computations to avoid overflow
- Adaptive tuning during warmup phase
- Acceptance rate tracking

**Code Quality:**
- Type hints and documentation
- Vectorized operations for efficiency
- Proper random seed management for reproducibility

---

### Phase 3: Convergence Diagnostics

**Objective**: Verify that both algorithms reach the target posterior

**Diagnostics Applied:**
1. **Trace plots**: Visual inspection of mixing
2. **R-hat statistic**: Gelman-Rubin convergence diagnostic (target: < 1.01)
3. **Effective Sample Size (ESS)**: Accounting for autocorrelation
4. **Autocorrelation plots**: Assess mixing efficiency

**Configuration:**
- 4 chains with different initializations
- 10,000 iterations per chain
- 2,000 burn-in (warmup) iterations
- Final: 32,000 posterior samples (8,000 per chain)

---

### Phase 4: Algorithm Comparison

**Objective**: Compare Gibbs vs Metropolis-Hastings on multiple criteria

**Comparison Metrics:**

1. **Effective Sample Size (ESS)**
   - Gibbs: Higher ESS due to direct sampling
   - MH: Lower ESS due to rejections and autocorrelation

2. **Computational Efficiency (ESS per second)**
   - Measures practical performance on real hardware
   - Accounts for both speed and sample quality

3. **Convergence Speed**
   - How many iterations to reach R-hat < 1.01?
   - Which algorithm mixes faster?

4. **Acceptance Rates (MH only)**
   - Target: 23.4% (optimal for high-dimensional RWM)
   - Too high → small steps → high autocorrelation
   - Too low → many rejections → wasted computation

5. **Posterior Agreement**
   - Do both algorithms target the same distribution?
   - Compare posterior means and credible intervals

**Expected Results:**
- Gibbs: Higher ESS, faster convergence (when conjugacy available)
- MH: More general (works without conjugacy), requires tuning

---

### Phase 5: Data Application

**Dataset**: Medical insurance costs (1,337 observations)

**Model**: Simple Bayesian linear regression
- Response: Medical charges
- Predictors: Age, sex, BMI, children, smoker, region
- Priors: Weakly informative (data-driven inference)

**Note**: We use a **simple model intentionally** - this is a Monte Carlo Simulation course, not a modeling course. The focus is on algorithm mechanics, not model sophistication.

---

## Key Insights

### Why This is a Monte Carlo Simulation Project

1. **Algorithm Design**: Complete derivations from scratch
2. **Implementation**: Manual coding (no PyMC3, Stan, JAGS)
3. **Comparison**: Systematic evaluation of algorithm properties
4. **Understanding**: Deep knowledge of how/why MCMC works

### What We Demonstrate

✅ **Theoretical Understanding**: Full mathematical derivations  
✅ **Coding Skills**: Implementing algorithms from first principles  
✅ **Diagnostics**: Proper convergence assessment  
✅ **Critical Thinking**: Comparing trade-offs between methods  
✅ **Computational Efficiency**: ESS, timing, memory considerations  

### What We Don't Focus On

❌ Perfect model fit (not a modeling course)  
❌ Model selection via Bayes factors (not the main goal)  
❌ Complex hierarchical models (keep model simple)  
❌ Using black-box tools (defeats the learning purpose)  

---- **Error Variance (σ²)**: σ² ~ Inverse-Gamma(0.01, 0.01)
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

### Phase 4: Bayesian MCMC (Gibbs Sampling) - All Models
**Steps:**
1. Implement Gibbs sampler with conjugate priors
2. Run 4 parallel chains with 10,000 iterations each
3. Discard 2,000 warmup samples per chain
4. Generate trace plots for visual convergence assessment
5. Fit three models: baseline, log-transformed, and interaction

**Sampling Configuration:**
- Iterations per chain: 10,000
- Warmup (burn-in): 2,000
- Number of chains: 4
- Total posterior samples: 32,000 per model

**Results:**
All three models successfully converged:
- **Baseline Model**: Standardized charges as response
- **Log-Transformed Model**: log(charges) to handle skewness
- **Interaction Model**: log(charges) + smoker:bmi interaction

**Trace Plots**: All coefficients + σ² show excellent mixing across chains with no drift or poor convergence patterns for all models.

---

### Phase 5: Convergence Diagnostics (Person 3)
**Steps:**
1. Compute Autocorrelation Function (ACF) for all parameters
2. Calculate Effective Sample Size (ESS) for efficiency assessment
3. Verify rapid ACF decay and high ESS values
4. Compare convergence across all three models

**Results - Baseline Model:**

**Autocorrelation Analysis:**
- All parameters show rapid ACF decay to near-zero within 50 lags
- No significant long-range autocorrelation detected
- Indicates good mixing and independence of samples

**Effective Sample Size (ESS):**
```
Baseline Model:
  Mean ESS: 22,653 | Efficiency: 70.79%
  
Log-Transformed Model:
  Mean ESS: 22,400 | Efficiency: 70.00%
  
Interaction Model:
  Mean ESS: 21,800 | Efficiency: 68.13%
```

**Interpretation**: All models achieve >68% efficiency, meaning MCMC samples provide high-quality effectively independent samples for inference. This demonstrates excellent convergence across all model specifications.

---

### Phase 6: Bayesian vs Frequentist Comparison (Person 3)
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

### Phase 8: Posterior Predictive Checks (Person 2)
**Steps:**
1. Generate posterior predictive distribution (32,000 replicated datasets)
2. Compute 95% prediction intervals
3. Assess coverage: proportion of actual values within intervals
4. Calculate Bayesian RMSE

**Results - Baseline Model:**
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


## Expected Results

### Algorithm Performance Metrics

| Metric | Gibbs Sampling | Metropolis-Hastings | Winner |
|--------|---------------|---------------------|---------|
| **ESS (avg)** | ~25,000 | ~15,000 | Gibbs |
| **ESS per second** | ~2,500 | ~1,800 | Gibbs |
| **Acceptance rate** | 100% | 20-40% | Gibbs |
| **Convergence (iterations to R̂<1.01)** | ~500 | ~800 | Gibbs |
| **Autocorrelation** | Lower | Higher | Gibbs |
| **Generality** | Requires conjugacy | Works always | MH |
| **Tuning required** | None | Yes (proposal variance) | Gibbs |

### Key Takeaways

**Gibbs Sampling Advantages:**
- ✅ Higher effective sample size
- ✅ 100% acceptance (no rejections)
- ✅ Faster convergence with conjugate priors
- ✅ No tuning required

**Gibbs Sampling Limitations:**
- ❌ Requires conjugate priors (mathematical constraints)
- ❌ Not applicable to all models

**Metropolis-Hastings Advantages:**
- ✅ Works for any posterior distribution
- ✅ Flexible proposal design
- ✅ Can handle non-conjugate models

**Metropolis-Hastings Limitations:**
- ❌ Requires careful tuning of proposal variance
- ❌ Lower effective sample size due to rejections
- ❌ Higher autocorrelation

### Why Both Matter

This comparison demonstrates a fundamental trade-off in MCMC design:
- **Specialized algorithms** (Gibbs): Fast and efficient but limited scope
- **General algorithms** (MH): Broadly applicable but require more tuning

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