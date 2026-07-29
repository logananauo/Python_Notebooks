### K-FOLD CROSS VALIDATION WITH NUMPY ARRAY ###


import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


### synthetic regression data
X, y = make_regression(
    n_samples=1000,
    n_features=1,
    noise=10,
    random_state=42
)


### pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])


### kfold
kf = KFold(n_splits=5, shuffle=False)

rmse_scores = []

for train_index, val_index in kf.split(X):

    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # train model
    pipeline.fit(X_train, y_train)

    # make predictions
    y_pred = pipeline.predict(X_val)

    # calculate rmse for current fold
    fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(fold_rmse)

# calculate and display the final rmse score
average_rmse = np.mean(rmse_scores)
print("Average RMSE:", average_rmse)