import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.multivariate_ols import MultivariateLS
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
df = pd.read_csv('Lego_MLR.csv', sep = '\t')

x_1 = df["Pieces"]
x_2 = df["Size"]
y = df["Amazon_Price"]

n = len(df["Size"])
print(n)
i = 0
for i in range(n):
# this loop will make a new column with 0 for small and 1 for large size Lego sets
    size_num = df["Size"][i]
    if df["Size"][i] == Large:
        df["Size_1"][i] = 1
    else:
        df["Size_1"][i] = 0
    i += 1

beta_zero = 0.0
beta_one = 0.0
beta_two = 0.0
eps = 0.0  #episilon

    ## Part 2 ##
# Using model 1
x1_bar = np.mean(x_1)
y_bar = np.mean(y)

Sxx = np.sum((x_1 - x1_bar)**2)
Sxy = np.sum((x_1 - x1_bar)*(y - y_bar))
Syy = np.sum((y - y_bar)**2)

beta1_hat = Sxy/Sxx
beta0_hat = y_bar - beta1_hat * x1_bar

print("Sxx =", Sxx)
print("Sxy =", Sxy)
print("Syy =", Syy)

# Equation for when Model 1 has xi2 = 0.0
y_small = beta_zero + beta_one*x_1 + beta_two*0.0 + eps

# Equation for when Model 1 has  xi2 = 1.0
y_large = beta_zero + beta_one*x_1 + beta_two*1.0+ eps



    ## Part 3 ##
# Using Model 2

# Equation for when Model 2 has xi2 = 0.0
xi_2 = 0.0
y_small = beta_zero + beta_one*x_1 + beta_two*0.0 + beta_12*xi_1*xi2+ eps

# Equation for when Model 1 has  xi2 = 1.0
xi_2 = 1.0
y_large = beta_zero + beta_one*x_1 + beta_two*0.0 + beta_12*xi_1*xi2+ eps

    ## Part 4, run linear regression for model 1 and model 2
model = smf.ols('Amazon_Price ~ Pieces + Size', data=df).fit()
model.summary()
