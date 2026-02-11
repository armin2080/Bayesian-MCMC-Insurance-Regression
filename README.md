# Bayesian MCMC Insurance Regression

This project implements Bayesian linear regression using MCMC methods to analyze health insurance charges data. The analysis compares Gibbs sampling and Metropolis-Hastings algorithms for parameter estimation.

## Project Structure

- **data/** - Insurance dataset (raw and cleaned)
- **python/** - Python implementation
  - `scripts/` - Core analysis scripts
  - `outputs/` - Generated results and plots
  - `requirements.txt` - Python dependencies
- **r/** - R implementation
  - `scripts/` - R analysis scripts
  - `outputs/` - R-generated results
- **Final_Report/** - LaTeX report and bibliography
- **env/** - Python virtual environment

## Implementation

The project uses 4 MCMC chains with 10,000 iterations (2,000 burn-in) to estimate regression parameters. Key features include:

- Gibbs sampling with conjugate priors
- Metropolis-Hastings with adaptive proposals
- Convergence diagnostics (Gelman-Rubin, ESS)
- Posterior predictive checks
- Algorithm efficiency comparison

## Requirements

**Python:**
- Python 3.12+
- numpy, scipy, matplotlib, pandas

**R:**
- R 4.0+
- coda, ggplot2, dplyr

## Usage

**Python analysis:**
```bash
source env/bin/activate
cd python/scripts
python run_full_analysis.py
```

**R analysis:**
```r
source("r/scripts/Run_Full_Analysis.R")
```

## Results

Generated outputs include:
- Posterior distributions
- Trace plots and ACF plots
- Convergence statistics (R-hat, ESS)
- Model comparison metrics
- Residual diagnostics

See `Final_Report/Report.pdf` for complete analysis.
