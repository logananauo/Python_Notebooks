### TRAIN-TEST SPLIT PIPELINE ###

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


### define tests
mytest_list = [0.40, 0.30, 0.20, 0.10, 0.05]
mytest_len = len(mytest_list)


### define pipeline funtion to run continuously
def run_pipeline(n_runs=5):
    rmse_list = []
    test_list = []

    for i in range(n_runs):
        mytest = mytest_list[i % mytest_len]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=mytest, random_state=2120)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
        
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        rmse_list.append(rmse)
        test_list.append(mytest)

        print(f'Run {i+1}: test = {mytest}, RMSE = {rmse:.2f}')

    print('\n')
    print(f'Average RMSE: {np.mean(rmse_list)}')
    print(f'Standard Deviation of RMSE values: {np.std(rmse_list)}')
    return pd.DataFrame({
        'test': test_list,
        'RMSE': rmse_list
    })


### display results
results_df = run_pipeline(n_runs=5)
results_df