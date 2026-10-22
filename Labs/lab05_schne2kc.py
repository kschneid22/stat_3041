import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.multivariate_ols import MultivariateLS
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
df = pd.read_csv('Lego_MLR.csv', sep = '\t')

x_1 = df["Pieces"]          # predictor variable, xi_1
x_2 = df["Size"]            # predictor variable, xi_2
y = df["Amazon_Price"]      # response variable, or y

# small = df[df['Size'] == 'Small'] #this creates a new data frame of just the small sets
# large = df[df['Size'] == 'Large'] #this creates a new data frame of just the large sets
small = df[df['Size'] == 'Small']
large = df[df['Size'] == 'Large']

plt.scatter(small['Pieces'], small['Amazon_Price'], facecolors='none', edgecolors='green', label='Small')
plt.scatter(large['Pieces'], large['Amazon_Price'], facecolors='none', edgecolors='blue', label='Large')

model = smf.ols('Amazon_Price ~ Pieces + Size', data=df).fit()
plt.plot(small['Pieces'], model.predict(small), color='green')
plt.plot(large['Pieces'], model.predict(large), color='blue')


plt.xlabel('Pieces')
plt.ylabel('Amazon Price')
plt.legend()
plt.show()

# betahat = (X^T X)^{-1}X^T Y
# Design matrix: intercept, Size (1 = Large, 0 = Small), Pieces
X = np.column_stack([np.ones(len(df)),                     (df['Size'] == 'Large').astype(int), #It is similar to Python’s int(), float() and str() functions,
                                                        #which convert one value at a time, but astype converts a whole column at once.
                     df['Pieces']])
y = df['Amazon_Price'].to_numpy()

# beta_hat = (X'X)^{-1} X'y
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
print(beta_hat)

# compare with statsmodels
model = smf.ols('Amazon_Price ~ Pieces + Size', data=df).fit()
print(model.params.values)
beta_zero = 0.0
# beta_one = 0.0
# beta_two = 0.0
# eps = 0.0  #episilon
#
#     ## Part 2 ##
# # Using model 1
# x1_bar = np.mean(x_1)
# y_bar = np.mean(y)
#
# Sxx = np.sum((x_1 - x1_bar)**2)
# Sxy = np.sum((x_1 - x1_bar)*(y - y_bar))
# Syy = np.sum((y - y_bar)**2)
#
# beta1_hat = Sxy/Sxx
# beta0_hat = y_bar - beta1_hat * x1_bar
#
# print("Sxx =", Sxx)
# print("Sxy =", Sxy)
# print("Syy =", Syy)
#
# # Equation for when Model 1 has xi2 = 0.0
# y_small = beta_zero + beta_one*x_1 + beta_two*0.0 + eps
#
# # Equation for when Model 1 has  xi2 = 1.0
# y_large = beta_zero + beta_one*x_1 + beta_two*1.0+ eps
#
#
#
#     ## Part 3 ##
# # Using Model 2
#
# # Equation for when Model 2 has xi2 = 0.0
# xi_2 = 0.0
# y_small = beta_zero + beta_one*x_1 + beta_two*0.0 + beta_12*xi_1*xi2+ eps
#
# # Equation for when Model 1 has  xi2 = 1.0
# xi_2 = 1.0
# y_large = beta_zero + beta_one*x_1 + beta_two*0.0 + beta_12*xi_1*xi2+ eps

    ## Part 4, run linear regression for model 1 and model 2

model1 = smf.ols('Amazon_Price ~ Pieces + Size',data=df).fit()

print("MODEL 1")
print(model1.summary())

print("\nModel 1 Coefficients:")
print(model1.params)
model2 = smf.ols('Amazon_Price ~ Pieces * Size', data=df).fit()

print("\n\nMODEL 2")
print(model2.summary())

print("\nModel 2 Coefficients:")
print(model2.params)
model2 = smf.ols(
    'Amazon_Price ~ Pieces * Size',
    data=df
).fit()

# Separate data by size
small = df[df['Size'] == 'Small'].copy()
large = df[df['Size'] == 'Large'].copy()

# Sort for smooth regression lines
small = small.sort_values('Pieces')
large = large.sort_values('Pieces')

# Scatter plot of actual data
plt.figure(figsize=(8,6))

plt.scatter(    small['Pieces'],
    small['Amazon_Price'],
    facecolors='none',
    edgecolors='green',
    label='Small Sets'
)

plt.scatter(
    large['Pieces'],    large['Amazon_Price'],
    facecolors='none',
    edgecolors='blue',
    label='Large Sets'
)

# Predicted regression lines from Model 2
plt.plot(
    small['Pieces'],
    model2.predict(small),
    color='green',
    linewidth=2,
    label='Small MLR Line'
)

plt.plot(
    large['Pieces'],
    model2.predict(large),
    color='blue',
    linewidth=2,
    label='Large MLR Line'
)

plt.xlabel('Pieces')
plt.ylabel('Amazon Price (USD)')
plt.title('Price vs Pieces with Size Interaction')
plt.legend()
plt.grid(True)

plt.show()