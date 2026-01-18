"""
Quick Test Script for Python Implementation

This script runs a quick test to verify all modules are working correctly.
"""

import sys
import numpy as np

def test_imports():
    """Test if all required modules can be imported."""
    print("Testing imports...")
    try:
        import pandas
        print("  ✓ pandas")
        import numpy
        print("  ✓ numpy")
        import matplotlib.pyplot
        print("  ✓ matplotlib")
        import seaborn
        print("  ✓ seaborn")
        import scipy
        print("  ✓ scipy")
        import sklearn
        print("  ✓ scikit-learn")
        import statsmodels
        print("  ✓ statsmodels")
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_custom_modules():
    """Test if custom modules can be imported."""
    print("\nTesting custom modules...")
    try:
        from data_preprocessing import preprocess_data
        print("  ✓ data_preprocessing")
        
        from gibbs_sampling import gibbs_lm
        print("  ✓ gibbs_sampling")
        
        from convergence_detection import acf_plot_beta
        print("  ✓ convergence_detection")
        
        from posterior_inference import posterior_predictive
        print("  ✓ posterior_inference")
        
        return True
    except ImportError as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_gibbs_sampler():
    """Test the Gibbs sampler with synthetic data."""
    print("\nTesting Gibbs sampler...")
    try:
        from gibbs_sampling import gibbs_lm
        
        # Generate synthetic data
        np.random.seed(42)
        n = 50
        p = 3
        
        X = np.column_stack([np.ones(n), np.random.randn(n, p-1)])
        true_beta = np.array([2.0, 1.5, -1.0])
        y = X @ true_beta + np.random.randn(n) * 0.5
        
        # Run short Gibbs sampler
        results = gibbs_lm(y, X, n_iter=100, warmup=20, n_chains=2, seed=42)
        
        if len(results) == 2:
            print("  ✓ Gibbs sampler executed successfully")
            print(f"    - Generated {len(results)} chains")
            print(f"    - Beta samples shape: {results[0]['beta'].shape}")
            return True
        else:
            print("  ✗ Unexpected output from Gibbs sampler")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_data_file():
    """Test if the data file exists."""
    print("\nTesting data file...")
    from pathlib import Path
    
    data_path = Path('../../data/expenses.csv')
    if data_path.exists():
        print(f"  ✓ Data file found: {data_path}")
        return True
    else:
        print(f"  ✗ Data file not found: {data_path}")
        print("    Please ensure expenses.csv is in the data/ directory")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("PYTHON IMPLEMENTATION - QUICK TEST")
    print("="*70)
    
    results = []
    
    # Test imports
    results.append(test_imports())
    
    # Test custom modules
    results.append(test_custom_modules())
    
    # Test Gibbs sampler
    results.append(test_gibbs_sampler())
    
    # Test data file
    results.append(test_data_file())
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    if all(results):
        print("\n✅ All tests passed! The Python implementation is ready to use.")
        print("\nTo run the complete analysis:")
        print("  python model_setup.py")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
    
    print("\n" + "="*70)
    
    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
