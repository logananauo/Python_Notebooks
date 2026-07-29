### REGRESSION METRICS ###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


### sythetic regression dataset
X, y = make_regression(n_samples=1000, # rows of data
                       n_features=1, # how many x variables
                       noise=20, # how much error to introduce
                       bias=200, # what is the intercept
                       random_state=42) # reproduceability


### convert to a dataframe
if (X.shape[1] == 1):
    feature_names =['X']
else:
    feature_names = [f'X{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['y'] = y
df.head()


### visualize
plt.scatter(X, y)
plt.xlabel('Predictor Variable (X)')
plt.ylabel('Response Variable (y)')
plt.title('Synthetic Regression Data')
plt.show()


### fit linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
residuals = y-y_pred


### visualize results
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')
plt.show()


### calculate evaluation metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)
n = len(y)
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)


### create dataframe to display
metrics_df = pd.DataFrame({
    'Metric': ['MAE', 'MSE', 'RMSE', 'R2', 'Adjusted R2'],
    'Value': [mae, mse, rmse, r2, adj_r2]
})
metrics_df
