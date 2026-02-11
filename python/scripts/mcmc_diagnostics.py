import numpy as np

def rhat(chains_2d):
    """
    Gelman-Rubin R-hat diagnostic for convergence.
    Pass in shape (n_chains, n_samples).
    """
    x = np.asarray(chains_2d)
    if x.ndim != 2:
        raise ValueError("rhat expects shape (m_chains, n_samples)")

    m, n = x.shape
    chain_means = x.mean(axis=1)
    grand_mean = chain_means.mean()

    B = n * np.sum((chain_means - grand_mean) ** 2) / (m - 1)
    W = np.sum(x.var(axis=1, ddof=1)) / m
    var_hat = ((n - 1) / n) * W + (1 / n) * B

    return np.sqrt(var_hat / W)


def autocovariance_1d(x):
    """Naive O(n^2) autocovariance, good enough for our purposes."""
    x = np.asarray(x)
    n = x.size
    x = x - x.mean()
    acov = np.empty(n)
    for k in range(n):
        acov[k] = np.dot(x[:n-k], x[k:]) / (n - k)
    return acov


def ess_geyer(chains_2d):
    """
    Effective sample size via Geyer's initial positive sequence method.
    Pools chains and uses autocorrelation to estimate how many independent samples we have.
    """
    x = np.asarray(chains_2d)
    if x.ndim != 2:
        raise ValueError("ess_geyer expects shape (m_chains, n_samples)")

    m, n = x.shape
    pooled = x.reshape(-1)
    N = pooled.size

    acov = autocovariance_1d(pooled)
    if acov[0] <= 0:
        return np.nan

    rho = acov / acov[0]

    # sum consecutive pairs until they turn negative
    t = 1
    s = 0.0
    while (t + 1) < len(rho):
        pair_sum = rho[t] + rho[t + 1]
        if pair_sum < 0:
            break
        s += pair_sum
        t += 2

    tau = 1.0 + 2.0 * s
    return N / tau
