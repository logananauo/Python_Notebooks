### MAHCINE LEARNING PIPELINE WITH DIFFERENT SEEDS ###

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


### define seeds
myseed_list = [1, 7, 90, 170, 457, 3489, 5409, 6700, 9999, 10798]
myseed_len = len(myseed_list)


### for pandas dataframe
#X = df[['X']] # can hold multiple x variables in a multivariate case
#y = df['y']


### function to run pipeline repeatedly
def run_pipeline(n_runs=10):
    rmse_list = []
    r2_list = []
    seed_list = []

    for i in range(n_runs):
        # select new seed each run
        myseed = myseed_list[i % myseed_len]

        # split the data differently each time
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=myseed)

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
        seed_list.append(myseed)

        print(f'Run {i+1}: seed = {myseed}, RMSE = {rmse:.2f}, R2 = {r2:.2f}')

    return pd.DataFrame({
        'seed': seed_list,
        'RMSE': rmse_list,
        'R2': r2_list
    })


### run the pipeline 10 times and show the results
results_df = run_pipeline(n_runs=10)
results_df