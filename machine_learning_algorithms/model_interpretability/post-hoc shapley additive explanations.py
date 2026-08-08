#!pip install shap

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# enable JS visualization in notebooks
#shap.initjs()


### generate synthetic regression dataset
X, y = make_regression(n_samples=500, 
                       n_features=5, 
                       n_informative=3, 
                       noise=0.2, 
                       random_state=42)


### to dataframe
feature_names = [f'X{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)


### train-test split 
X_train, X_test, y_train, y_test = train_test_split(X_df, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)


### train a tree-based model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


### create SHAP explainer and compute SHAP values
explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)


### Global explanation - feature importance
shap.plots.bar(shap_values)


### Local explanation - individual prediction
#### pick a specific sample from the test set
sample_index = 0
shap.plots.waterfall(shap_values[sample_index])


