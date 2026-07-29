### BAGGING FOR REGRESSION ###

from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor


### synthetic regression dataset
X, y = make_regression(n_samples=500, 
                       n_features=5, 
                       n_informative=2, 
                       noise=30, 
                       bias=100, 
                       random_state=42)


### train-test split on input data 75/25
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.25, 
                                                    random_state=42)


### base learner 
base_tree = DecisionTreeRegressor(random_state=0)


### bagging model 
bag = BaggingRegressor(estimator=base_tree,
                       n_estimators=10,
                       bootstrap=True, 
                       random_state=0)


#NOTE: bootstrapping happens internally within BaggingRegressor
# it is important to leave some data completely unseen (i.e. the test data)
# in order to properly evaluate the performance


### fit both models
base_model = base_tree.fit(X_train, y_train)
bag_model = bag.fit(X_train, y_train)


### compare single tree vs bagging performance
single_pred = base_model.predict(X_test)
ensemble_pred = bag_model.predict(X_test)

print('MSE of Single Tree Model on test dataset: ', mean_squared_error(y_test, single_pred))
print('MSE of Bagging 10 Trees on test dataset: ', mean_squared_error(y_test, ensemble_pred))
