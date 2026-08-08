
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split


### generate synthetic regression dataset
X, y = make_regression(n_samples=1000, 
                       n_features=3, 
                       n_informative=3, 
                       noise=10, 
                       random_state=42)


### to dataframe
feature_names = ['X1', 'X2', 'X3']
X = pd.DataFrame(X, columns=feature_names)


### train-test split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)


### train a random forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


### plot partial dependence for individual features
features_to_plot = ['X1', 'X2', 'X3']
PartialDependenceDisplay.from_estimator(model, X_test, features=features_to_plot)
plt.suptitle('Partial Dependence Plots (Random Forest)', fontsize=16)
plt.tight_layout()
plt.show()