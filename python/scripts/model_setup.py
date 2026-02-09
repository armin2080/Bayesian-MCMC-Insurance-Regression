import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_preprocessing import preprocess_data
from gibbs_sampling import (gibbs_lm, beta_trace_plot, sigma2_trace_plot,
                            beta_summary_stats, sigma2_summary_stats)
from convergence_detection import (acf_plot_beta, acf_plot_sigma2,
                                   ess_beta_table, ess_sigma2_table)
from posterior_inference import (posterior_predictive, ppc_plot,
                                 PPC_density_overlay, ppc_residual_plot)


def create_design_matrix(df, formula='baseline'):
    # Select features
    if formula == 'baseline':
        # charges ~ age + sex + bmi + children + smoker
        feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker']
        y = df['charges'].values
    elif formula == 'log':
        # log(charges) ~ age + sex + bmi + children + smoker
        feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker']
        y = np.log(df['charges'].values)
    elif formula == 'interaction':
        # log(charges) ~ age + sex + bmi + children + smoker + smoker:bmi
        feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'smoker_bmi']
        # Create interaction term
        df_copy = df.copy()
        df_copy['smoker_bmi'] = df_copy['smoker'] * df_copy['bmi']
        y = np.log(df_copy['charges'].values)
        # Build design matrix
        X_data = df_copy[feature_cols].values
        X = np.column_stack([np.ones(len(df_copy)), X_data])
        feature_names = ['(Intercept)'] + feature_cols
        return X, y, feature_names
    else:
        raise ValueError(f"Unknown formula: {formula}")
    
    # Build design matrix
    X_data = df[feature_cols].values
    X = np.column_stack([np.ones(len(df)), X_data])
    feature_names = ['(Intercept)'] + feature_cols
    
    return X, y, feature_names


def run_ols_baseline(df):
    """
    Frequentist OLS baseline with correct intercept handling + standard errors.
    Uses statsmodels for clean inference output (coef, se, t, p, CI).
    """
    print("\n" + "="*70)
    print("BASELINE OLS REGRESSION (FREQUENTIST)")
    print("="*70)

    import statsmodels.api as sm

    # Your design matrix already includes intercept column (ones)
    X, y, feature_names = create_design_matrix(df, formula="baseline")

    # statsmodels OLS does NOT add an intercept unless you do sm.add_constant.
    # Since your X already contains the intercept column, we use it as-is.
    ols_model = sm.OLS(y, X).fit()

    print("\nCoefficients (OLS):")
    for name, coef, se, pval in zip(feature_names, ols_model.params, ols_model.bse, ols_model.pvalues):
        print(f"{name:>12s}  coef={coef: .6f}  se={se: .6f}  p={pval: .4g}")

    print(f"\nR^2 = {ols_model.rsquared:.4f}")
    print(f"Adj R^2 = {ols_model.rsquared_adj:.4f}")

    # Return a dict compatible with your pipeline
    return {
        "params": ols_model.params,
        "bse": ols_model.bse,
        "tvalues": ols_model.tvalues,
        "pvalues": ols_model.pvalues,
        "rsquared": ols_model.rsquared,
        "rsquared_adj": ols_model.rsquared_adj,
        "resid": ols_model.resid,
        "fittedvalues": ols_model.fittedvalues
    }


def compute_correlation_matrix(df):
    print("\n" + "="*70)
    print("CORRELATION MATRIX")
    print("="*70)
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    print("\n", corr_matrix)
    
    # Save to file
    output_path = Path('../../r/outputs/cor_matrix.txt')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corr_matrix.to_csv(output_path, sep='\t', float_format='%.4f')
    
    print(f"\nCorrelation matrix saved to: {output_path}")


def run_bayesian_model(df, model_name, formula='baseline'):
    print("\n" + "="*70)
    print(f"BAYESIAN MODEL: {model_name.upper()}")
    print("="*70)
    
    # Prepare data
    X, y, feature_names = create_design_matrix(df, formula=formula)
    
    print(f"\nModel formula: {formula}")
    print(f"Response transformation: {'log(charges)' if formula != 'baseline' else 'charges'}")
    print(f"Design matrix shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    # ==========================================
    # 1. Run Gibbs Sampler
    # ==========================================
    print("\n" + "-"*70)
    print("Step 1: Running Gibbs Sampler...")
    print("-"*70)
    
    results = gibbs_lm(
        y=y,
        X=X,
        n_iter=10000,
        warmup=2000,
        n_chains=4,
        seed=123
    )
    
    beta_list = [chain['beta'] for chain in results]
    sigma2_list = [chain['sigma2'] for chain in results]
    
    # ==========================================
    # 2. Trace Plots
    # ==========================================
    print("\n" + "-"*70)
    print("Step 2: Creating Trace Plots...")
    print("-"*70)
    
    beta_trace_plot(beta_list, model_name)
    sigma2_trace_plot(sigma2_list, model_name)
    
    # ==========================================
    # 3. Summary Statistics
    # ==========================================
    print("\n" + "-"*70)
    print("Step 3: Computing Summary Statistics...")
    print("-"*70)
    
    beta_summary = beta_summary_stats(beta_list)
    sigma2_summary = sigma2_summary_stats(sigma2_list)
    
    # ==========================================
    # 4. Convergence Diagnostics - ACF
    # ==========================================
    print("\n" + "-"*70)
    print("Step 4: Computing Autocorrelation Functions...")
    print("-"*70)
    
    acf_plot_beta(beta_list, model_name)
    acf_plot_sigma2(sigma2_list, model_name)
    
    # ==========================================
    # 5. Convergence Diagnostics - ESS
    # ==========================================
    print("\n" + "-"*70)
    print("Step 5: Computing Effective Sample Sizes...")
    print("-"*70)
    
    # Create DataFrame for X with feature names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    ess_beta_table(beta_list, X_df, model_name)
    ess_sigma2_table(sigma2_list, model_name)
    
    # ==========================================
    # 6. Posterior Predictive Checks
    # ==========================================
    print("\n" + "-"*70)
    print("Step 6: Posterior Predictive Checks...")
    print("-"*70)
    
    y_rep = posterior_predictive(
        beta_list=beta_list,
        sigma2_list=sigma2_list,
        X_new=X
    )
    
    ppc_plot(y, y_rep, model_name)
    PPC_density_overlay(y, y_rep, model_name)
    ppc_residual_plot(y, y_rep, model_name)
    
    print(f"\n{model_name.upper()} ANALYSIS COMPLETE!")
    print("="*70)


def main():
    print("\n" + "="*70)
    print("BAYESIAN MCMC INSURANCE REGRESSION")
    print("Complete Analysis Pipeline")
    print("="*70)
    
    # ==========================================
    # STEP 1: Data Preprocessing
    # ==========================================
    print("\n" + "="*70)
    print("STEP 1: DATA PREPROCESSING")
    print("="*70)
    
    df_clean = preprocess_data(
        input_path='../../data/expenses.csv',
        output_path='../../data/expenses_cleaned.csv'
    )
    
    # ==========================================
    # STEP 2: Baseline Analysis
    # ==========================================
    print("\n" + "="*70)
    print("STEP 2: BASELINE ANALYSIS")
    print("="*70)
    
    ols_results = run_ols_baseline(df_clean)
    compute_correlation_matrix(df_clean)
    
    # ==========================================
    # STEP 3: Bayesian Model 1 - Baseline
    # ==========================================
    run_bayesian_model(df_clean, 'gibbs_result', formula='baseline')
    
    # ==========================================
    # STEP 4: Bayesian Model 2 - Log Transform
    # ==========================================
    run_bayesian_model(df_clean, 'log_fit', formula='log')
    
    # ==========================================
    # STEP 5: Bayesian Model 3 - Interaction
    # ==========================================
    run_bayesian_model(df_clean, 'interaction', formula='interaction')
    
    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    print("\n" + "="*70)
    print("COMPLETE ANALYSIS FINISHED!")
    print("="*70)
    print("\nAll models have been successfully fitted and analyzed.")
    print("\nOutputs:")
    print("  - Trace plots: ../../plots/{model_name}/")
    print("  - ACF plots: ../../plots/{model_name}/ACF_plots/")
    print("  - PPC plots: ../../plots/{model_name}/PPC/")
    print("  - ESS tables: ../../r/outputs/{model_name}/ESS_tables/")
    print("  - OLS results: ../../r/outputs/ols_output.txt")
    print("  - Correlation matrix: ../../r/outputs/cor_matrix.txt")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
