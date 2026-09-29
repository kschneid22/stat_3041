import numpy as np
from scipy.stats import norm

def test_coverage(n, true_p, method="population_sd"):
    confidence = 0.95
    B = 100_000
    count_contains = 0
    z = norm.ppf(1 - (1 - confidence) / 2)
    for i in range(B):
        sample = np.random.exponential(1, size=n)
        sample_mean = np.mean(sample)
        if method == "population_sd":
            se = np.sqrt(true_p * (1 - true_p) / n)
        elif method == "sample_sd":
            se = np.std(sample, ddof=1) / np.sqrt(n)
        elif method == "plugin_p":
            se = np.sqrt(sample_mean * (1 - sample_mean) / n)
        lower = sample_mean - z * se
        upper = sample_mean + z * se
        if lower <= true_p <= upper:
            count_contains += 1
    coverage = count_contains/B
    return coverage

ss_5 = test_coverage(5,.3,method="sample_sd")
ss_10 = test_coverage(10,.3,method="sample_sd")
ss_30 = test_coverage(30,.3,method="sample_sd")
ss_50 = test_coverage(50,.3,method="sample_sd")
ss_100 = test_coverage(100,.3,method="sample_sd")


pop_5 = test_coverage(5,.3,method="population_sd")
pop_10 = test_coverage(10,.3,method="population_sd")
pop_30 = test_coverage(30,.3,method="population_sd")
pop_50 = test_coverage(50,.3,method="population_sd")
pop_100 = test_coverage(100,.3,method="population_sd")



plug_5 = test_coverage(5,.3,method="plugin_p")
plug_10 = test_coverage(10,.3,method="plugin_p")
plug_30 = test_coverage(30,.3,method="plugin_p")
plug_50 = test_coverage(50,.3,method="plugin_p")
plug_100 = test_coverage(100,.3,method="plugin_p")

print (' w population sd n = 5 coverage is,' , pop_5)
print ('w population sd n = 10 coverage is,' , pop_10)
print ('w population sd n = 30 coverage is,' , pop_30)
print (' w population sd n = 50 coverage is,' , pop_50)
print (' w population sd n = 100 coverage is,' , pop_100)

print('XXXXXXXXXXXXXXXXXXXXXXXXXXX')

print (' w sample sd n = 5 coverage is,' , ss_5)
print ('w sample sd n = 10 coverage is,' , ss_10)
print ('w sample sd n = 30 coverage is,' , ss_30)
print (' w sample sd n = 50 coverage is,' , ss_50)
print (' w sample sd n = 100 coverage is,' , ss_100)

print('XXXXXXXXXXXXXXXXXXXXXXXXXXX')

print(' w plug sd n = 5 coverage is,' , plug_5)
print(' w plug sd n = 10 coverage is,' , plug_10)
print(' w plug sd n = 30 coverage is,' , plug_30)
print(' w plug sd n = 50 coverage is,' , plug_50)
print(' w plug sd n = 100 coverage is,' , plug_100)
