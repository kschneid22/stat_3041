import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.multivariate_ols import MultivariateLS
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sklearn
from sklearn.metrics import roc_curve, roc_auc_score
from math import comb


## import heart disease data frame
df = pd.read_csv("Heart_data.csv")


## patients with and without heart disease, describes the continuous data in the df
print(df['diagnosis'].value_counts())      # patients with and without breast cancer
print(df[['thalach', 'age', 'trestbps', 'chol',]].describe())


#print least squares of diagnoses against thalach amounts
model = smf.ols('diagnosis ~ thalach', data=df).fit()
grid = pd.DataFrame({'thalach': np.linspace(50.0, 205, 100)})

plt.scatter(df['thalach'], df['diagnosis'], facecolors='none', edgecolors='blue')
plt.plot(grid['thalach'], model.predict(grid), color='red', label='Least squares line')
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(1, color='gray', linestyle='--')
plt.xlabel('thalach')
plt.ylabel('diagnosis (0 = benign, 1 = malignant)')
plt.legend()
plt.show()


model1 = smf.logit('diagnosis ~ thalach + age + sex + angina + non_angina + asymptomatic', data=df).fit()
model2 = smf.logit('diagnosis ~ thalach + age + trestbps + chol  + sex + angina + non_angina + asymptomatic', data=df).fit()
print(model.summary())




# Prediction
new = pd.DataFrame({'thalach': [140], 'age': [140],'trestbps': [130], 'chol': [240],'sex': [1],'angina': [0],'non_angina' : [0], 'asymptomatic': [1]})
print(model1.predict(new)[0])

## cutoff table
df['p_hat'] = model1.predict(df)
df['pred'] = (df['p_hat'] > 0.5).astype(int) # You can change to 0.5, 0.2, etc

table = pd.crosstab(df['diagnosis'], df['pred'],  rownames=['Actual'], colnames=['Predicted'])
table = table.reindex(index=[1, 0], columns=[1, 0])
print(table)

## accuracy
p_hat = model1.predict(df)                       # predicted P(malignant)

fpr, tpr, thresholds = roc_curve(df['diagnosis'], p_hat)
auc = roc_auc_score(df['diagnosis'], p_hat)

plt.plot(fpr, tpr, color='blue', label=f'Logistic regression (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random guessing')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve, without trestbps & chol')
plt.legend()
plt.show()

print("model", 1, "AIC: ", model1.aic, "BIC: ", model1.bic)
print("model", 2, "AIC: ", model2.aic, "BIC: ", model2.bic)