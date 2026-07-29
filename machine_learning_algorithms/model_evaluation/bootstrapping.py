### BOOTSTRAPPING ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.utils import resample


### synthetic regression data
X, y = make_regression(n_samples=1000, 
                       n_features=1, 
                       noise=10, 
                       random_state=42)

df = pd.DataFrame({'X': X.flatten(), 'y': y})
df.head()


### visualize
plt.figure(figsize=(16, 9))
plt.scatter(df['X'], df['y'], alpha=0.7)
plt.title('Synthetic Regression Data')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.show()


### train a linear regression model on bootstrap samples and evaluate oob
n_iterations = 1000
n_size = len(df)
rmse_scores = []
oob_sizes = []


### generate 1000 bootstrap samples, run linear regression, evaluate performance on OOB data
for boot in range(n_iterations):
    
    # sample with replacement
    sample = resample(df, n_samples=n_size, replace=True, random_state=boot)

    # OOB observations
    oob = df.loc[~df.index.isin(sample.index)]

    # train model
    model = LinearRegression()
    model.fit(sample[['X']], sample['y'])

    # predict and evaluate on OOB data
    if len(oob) > 0:
        y_pred = model.predict(oob[['X']])
        rmse = np.sqrt(mean_squared_error(oob['y'], y_pred))
        rmse_scores.append(rmse)
        oob_sizes.append(len(oob))


### summary statistics
mean_rmse = np.mean(rmse_scores)
std_rmse = np.std(rmse_scores)
ci_lower = np.percentile(rmse_scores, 2.5)
ci_upper = np.percentile(rmse_scores, 97.5)
mean_oob_size = np.mean(oob_sizes)
mean_oob_percent = 100 * mean_oob_size / n_size
print(f'Mean RMSE: {mean_rmse:.2f}')
print(f'Standard Deivation: {std_rmse:.2f}')
print(f'95% Confidence Interval: [{ci_lower:.2f}, {ci_upper:.2f}]')
print(f'Average oob percent: {mean_oob_percent:.2f}%')


### plot distribution of RMSE scores
plt.figure(figsize=(16, 9))
plt.hist(rmse_scores, bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of RMSE from Bootstrapping')
plt.xlabel('RMSE')
plt.ylabel('Frequency')
plt.axvline(mean_rmse, color='red', linestyle='dashed', linewidth=2, label=f'Mean RMSE = {mean_rmse:.2f}')
plt.legend()
plt.grid(True)
plt.show()
