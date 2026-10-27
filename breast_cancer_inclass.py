import pandas as pd

df = pd.read_csv("breast_cancer.csv")
df['diagnosis'] = df['diagnosis'].map({'malignant': 1, 'benign': 0}) #
df = df.rename(columns={'concave points': 'concave_points'})
print(df)

print(df['diagnosis'].value_counts())      # patients with and without breast cancer
print(df[['radius', 'concave_points', 'texture']].describe())

import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import pandas as pd

model = smf.ols('diagnosis ~ concave_points', data=df).fit()

grid = pd.DataFrame({'concave_points': np.linspace(-0.05, 0.35, 100)})

plt.scatter(df['concave_points'], df['diagnosis'], facecolors='none', edgecolors='blue')
plt.plot(grid['concave_points'], model.predict(grid), color='red', label='Least squares line')
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(1, color='gray', linestyle='--')
plt.xlabel('concave points')
plt.ylabel('diagnosis (0 = benign, 1 = malignant)')
plt.legend()
plt.show()


## least squares block
model = smf.logit('diagnosis ~ concave_points', data=df).fit()

grid = pd.DataFrame({'concave_points': np.linspace(-0.05, 0.35, 100)})

plt.scatter(df['concave_points'], df['diagnosis'], facecolors='none', edgecolors='blue')
plt.plot(grid['concave_points'], model.predict(grid), color='red', label='Logistic fit')
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(1, color='gray', linestyle='--')
plt.xlabel('concave points')
plt.ylabel('P(malignant)')
plt.legend()
plt.show()