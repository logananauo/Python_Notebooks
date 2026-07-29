### K-FOLD CROSS VALIDATION WITH PANDAS ###


import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error


### synthetic regression data
X, y = make_regression(n_samples=1000, 
                       n_features=1, 
                       noise=10, 
                       random_state=42)

X = pd.DataFrame(X, columns=['X'])
y = pd.Series(y)


### pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])


### kfold 
kf = KFold(n_splits=5, shuffle=False)

rmse_scores = []

for train_index, val_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    # reshape X to 2D arrays because scikit-learn expects 2D features
    #X_train_2d = X_train.values.reshape(-1, 1)
    #X_val_2d = X_val.values.reshape(-1, 1)
    
    # train model
    pipeline.fit(X_train, y_train)
    
    # make predictions
    y_pred = pipeline.predict(X_val)
    
    # calculate RMSE for the current fold
    fold_rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmse_scores.append(fold_rmse)

### calculate and display the final average RMSE
average_rmse = np.mean(rmse_scores)
print("Average RMSE score:", average_rmse)
