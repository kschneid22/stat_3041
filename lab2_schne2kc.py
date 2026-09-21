# Katrina Schneider, Lab 2 STAT3041, Sept 14 2026
import numpy as np
import matplotlib.pyplot as plt

"""Part 1"""
# Fixes the " starting point" of the random number generator , so that
# every time this code runs , it produces the exact same random numbers, makes our results  reproducible.
np.random.seed(0)

# The "population" of all customer wait times Maria could ever record
population  =  np.random.exponential(scale =3 ,  size = 100000)
true_mean  = np.mean(population)
true_std = np.std(population)

print(f"Population mean wait time : {true_mean:.4f} minutes") #printing this w limitedd number of sig figs since I hate when theyre long
print(f"Population std: {true_std: .4f} minutes" )

graphs = ['Original', 'Customized']

'''Original Histogram Block'''
plt.figure(figsize= (10,6))
plt.hist(population , density= True , bins= 100) # plot histogram
plt.title('Population Wait Time Histogram')
plt.xlabel('Wait (minutes)')
plt.ylabel('Density')
plt.grid(alpha=0.3)
plt.show()

'''Modified Histogram Block'''
plt.figure(figsize= (10,6))
plt.hist(population , density= True , bins= 50, color = 'pink') # plot histogram
plt.title('Population Wait Time Histogram, 50 bins')
plt.xlabel('Wait (minutes)')
plt.ylabel('Density')
plt.grid(alpha=0.3)
plt.show()


"""Part 2A Build 1 Confidence Interval"""
maria_30 = np.loadtxt("maria_30.txt")
print("30 sample shape is", maria_30.shape)
print("30 sample type is", type(maria_30))

sample_mean = np.mean(maria_30)
n = 30                              #count of customers samples
sigma = 3
z = 1.96                         #from a 95% confidence interval, assuming normal
lower_limit = sample_mean - z*(sigma/np.sqrt(n))
upper_limit = sample_mean + z*(sigma/np.sqrt(n))
print(f"Maria's 30 sample mean in, in minutes: {sample_mean:.4f}, 95% CI({lower_limit:.4f}, {upper_limit:.4f}) ")

"""Part 2B, 10,000 repeats of a sample size od 30"""
repeats = 10000
sample_means = []

for i in range(repeats):
    sample = np.random.exponential(scale=3, size=n)
    sample_means.append(np.mean(sample))

sample_means = np.array(sample_means)

plt.figure(figsize=(10, 6))
plt.hist(sample_means, bins=50, density=True, color="orange", edgecolor="black")
plt.title("Distribution of 10,000 Average Wait Times (n = 30)")
plt.xlabel("Sample Mean Wait Times (minutes)")
plt.ylabel("Density")
plt.grid(alpha=0.3)

# Add vertical lines
plt.axvline(sample_mean, color="red", linestyle="--", linewidth=2, label="Maria's Sample Mean")
plt.axvline(lower_limit, color="green", linestyle="-", linewidth=2, label="95% CI Lower")
plt.axvline(upper_limit, color="green", linestyle="-", linewidth=2, label="95% CI Upper")

plt.legend()
plt.show()

"""Part 3"""
contains_true_mean = 0 #counter for number of CI (out of 10,000) that contain the true mean of 3 minutes

for i in range(repeats):
    sample = np.random.exponential(scale=3, size=n)
    xbar = np.mean(sample)

    lower_ci = xbar - z * (sigma / np.sqrt(n)) #sigma  = 3, as before
    upper_ci = xbar + z * (sigma / np.sqrt(n))

    if 3 >= lower_ci and 3 <= upper_ci: #criteria to have 3 in the CI
        contains_true_mean += 1

coverage = contains_true_mean / repeats #this is a ratio

print(f"Empirical coverage over {repeats} intervals: {coverage:.4f}")
