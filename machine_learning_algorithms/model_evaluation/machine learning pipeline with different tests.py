### MACHINE LEARNING PIPELINE WITH DIFFERENT TESTS ###

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


### synthetic regression data
X, y = make_regression(n_samples=1000, 
                       n_features=1, 
                       noise=10, 
                       random_state=42)


### define tests; 50/50, 60/40, 70/30 etc.
mytest_list = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
mytest_len = len(mytest_list)


### for pandas dataframe
#X = df[['X']] # can hold multiple x variables in a multivariate case
#y = df['y']


### define function to run pipeline repeatedly
def run_pipeline(n_runs=10):
    rmse_list = []
    r2_list = []
    test_list = []

    for i in range(n_runs):
        # select new test each run
        mytest = mytest_list[i % mytest_len]

        # split the data differently each time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=mytest, random_state=33)

        # define pipeline: preprocessing + model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])

        # fit the pipeline
        pipeline.fit(X_train, y_train)

        # predict and evalute
        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        rmse_list.append(rmse)
        r2_list.append(r2)
        test_list.append(mytest)

        print(f'Run {i+1}: test = {mytest}, RMSE = {rmse:.2f}, R2 = {r2:.2f}')

    return pd.DataFrame({
        'test': test_list,
        'RMSE': rmse_list,
        'R2': r2_list
    })


### run the pipeline 10 times and show the results
results_df = run_pipeline(n_runs=10)
results_df