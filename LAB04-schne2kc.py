import numpy as np from scipy.stats import norm
def test_coverage(n, true_p, method="population_sd"):
    confidence = 0.95 B = 100_000
    count_contains = 0
    z = norm.ppf(1 - (1 - confidence) / 2)
    for i in range(B):
        sample = np.random.binomial(1, true_p, size=n) sample_mean = np.mean(sample)
        if method == "population_sd": se = np.sqrt(true_p * (1 - true_p) / n)
        elif method == "sample_sd": se = np.std(sample, ddof=1) / np.sqrt(n)
        elif method == "plugin_p": se = np.sqrt(sample_mean * (1 - sample_mean) / n)
        lower = sample_mean - z * se upper = sample_mean + z * se if lower <= true_p <= upper: count_contains += 1
        coverage = count_contains / B
    return coverage