### VOTING FOR REGRESSION ###

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


### synthetic regression dataset
X, y = make_regression(n_samples=500, 
                       n_features=5, 
                       n_informative=2, 
                       noise=30, 
                       bias=100, 
                       random_state=42)


### for pandas dataframes
#X = df[['X1', 'X2', 'X3', 'X4', 'X5']]
#y = df['y']


### train test split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25,
                                                    random_state=42)


### base learners
reg1 = LinearRegression()
reg2 = KNeighborsRegressor(n_neighbors=5)
reg3 = DecisionTreeRegressor(max_depth=5, random_state=42)


### create an averaging ensemble of these regressors
voting_reg = VotingRegressor([('lin', reg1), ('knn-5', reg2), ('dt', reg3)])


### evaluate mean squared error (MSE) on test set
for name, reg in [('LinearReg', reg1), ('KNN-5', reg2), ('DecisionTree', reg3), ('Ensemble', voting_reg)]:
    # fit model
    reg.fit(X_train, y_train)

    # generate predictions
    y_pred = reg.predict(X_test)

    # compute MSE/RMSE
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f'{name:12} test --RMSE: {rmse:.3f}')
