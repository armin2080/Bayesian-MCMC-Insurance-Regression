# Python Implementation - Bayesian MCMC Insurance Regression

Python implementation of Bayesian linear regression with Gibbs sampling.

## Structure

```
python/
├── scripts/              # Analysis scripts (matches r/scripts/)
│   ├── data_preprocessing.py
│   ├── gibbs_sampling.py
│   ├── convergence_detection.py
│   ├── posterior_inference.py
│   ├── model_setup.py
│   └── test_installation.py
├── outputs/              # Output directory
├── data_downloader.py    # Data download utility
├── notebook.ipynb        # Jupyter notebook
└── requirements.txt      # Dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run analysis
cd python/scripts
python model_setup.py

# Run tests
python test_installation.py
```

## Usage

```python
from data_preprocessing import preprocess_data
from gibbs_sampling import gibbs_lm
from convergence_detection import acf_plot_beta, ess_beta_table
from posterior_inference import posterior_predictive, ppc_plot

# Preprocess data
df = preprocess_data('../../data/expenses.csv', '../../data/expenses_cleaned.csv')

# Run Gibbs sampler
results = gibbs_lm(y, X, n_iter=10000, warmup=2000, n_chains=4)

# Diagnostics and plots
beta_list = [chain['beta'] for chain in results]
sigma2_list = [chain['sigma2'] for chain in results]

acf_plot_beta(beta_list, 'model_name')
ess_beta_table(beta_list, X, 'model_name')
```

## Models

1. **Baseline**: `charges ~ age + sex + bmi + children + smoker`
2. **Log-transformed**: `log(charges) ~ age + sex + bmi + children + smoker`
3. **Interaction**: `log(charges) ~ age + sex + bmi + children + smoker + smoker:bmi`

## Output

- Trace plots: `../../plots/{model_name}/`
- ACF plots: `../../plots/{model_name}/ACF_plots/`
- PPC plots: `../../plots/{model_name}/PPC/`
- ESS tables: `../../r/outputs/{model_name}/ESS_tables/`
