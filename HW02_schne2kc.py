## Katrina Schneider, STAT 3041
## Homework 2, 13 OCT 2026, Simple Linear Regression

import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load data
df = pd.read_csv('Lego_data.csv', sep = '\t')

# Predictor and response
x = df["Pieces"]
y = df["Amazon_Price"]

## Part 2

x_bar = np.mean(x)
y_bar = np.mean(y)

Sxx = np.sum((x - x_bar)**2)
Sxy = np.sum((x - x_bar)*(y - y_bar))
Syy = np.sum((y - y_bar)**2)

beta1_hat = Sxy / Sxx
beta0_hat = y_bar - beta1_hat * x_bar

print("Sxx =", Sxx)
print("Sxy =", Sxy)
print("Syy =", Syy)

print("beta0_hat =", beta0_hat)
print("beta1_hat =", beta1_hat)

##Part 3
model = smf.ols("Amazon_Price ~ Pieces", data=df).fit()
print(model.summary())

## Part 4

print("By-hand estimates: \n")
print("Intercept = ", beta0_hat)
print("Slope = ", beta1_hat)

print("Statsmodels estimates: ")
print(model.params)

## Question 5
residuals = model.resid

plt.figure(figsize=(8,5))
plt.scatter(x, residuals)
plt.axhline(0, color='red')
plt.xlabel("Pieces")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()

sm.qqplot(residuals, line='45')
plt.title("Normal Q-Q Plot")
plt.show()

## Part 6

print("Slope test p-value: \n")
print(model.pvalues["Pieces"])

## Part 8
sigma_hat = np.sqrt(model.mse_resid)
print("Residual Standard Error = ", sigma_hat)

## Part 9
x_grid = np.linspace(x.min(), x.max(), 100)
pred_df = pd.DataFrame({"Pieces": x_grid})

predictions = model.get_prediction(pred_df)
pred_summary = predictions.summary_frame(alpha=0.05)

plt.figure(figsize=(10, 6))

plt.scatter(x, y, alpha=0.7)

plt.plot(x_grid,  pred_summary["mean"],
    color="red", label="Regression Line"
)

# Confidence band
plt.fill_between(
    x_grid,pred_summary["mean_ci_lower"],
    pred_summary["mean_ci_upper"],
    color="blue",
    alpha=0.2,
    label="95% Confidence Band"
)

# Prediction band
plt.fill_between(
    x_grid,
    pred_summary["obs_ci_lower"],
    pred_summary["obs_ci_upper"],
    color="orange",
    alpha=0.2,
    label="95% Prediction Band"
)

plt.xlabel("Pieces")
plt.ylabel("Amazon Prices")
plt.legend()
plt.title("Regression with Confidence and Prediction Bands")
plt.show()

# Prediction interval at x*=95

new_x = pd.DataFrame({"Pieces": [95]})

pred95 = model.get_prediction(new_x)
print("Prediction for x *= 95 \n")
print(pred95.summary_frame(alpha=0.05))