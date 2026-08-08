
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


### generate synthetic regression dataset
X, y = make_regression(n_samples=1000, 
                       n_features=5, 
                       n_informative=3, 
                       noise=10, 
                       random_state=42)


## to dataframe
feature_names = [f'X{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y


### train-test split
X_train, X_test, y_train, y_test = train_test_split(df[feature_names], 
                                                    df['target'], 
                                                    test_size=0.2, 
                                                    random_state=42)


### train a random forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


### permutation feature importance
result = permutation_importance(rf, 
                                X_test, 
                                y_test, 
                                n_repeats=30, 
                                random_state=42, 
                                scoring='neg_mean_squared_error'
                               )


### show importance values
importances = pd.DataFrame({
    'Feature': feature_names, 
    'Importance Mean': result.importances_mean, 
    'Importance Std': result.importances_std, 
}).sort_values(by='Importance Mean', ascending=False)
print(importances)


### plot the importances
plt.figure(figsize=(8, 5))
plt.barh(importances['Feature'], 
         importances['Importance Mean'], 
         xerr=importances['Importance Std'], 
         color='skyblue')
plt.xlabel('Decrease in Model Performance\n(Negative MSE drop from permuting features)')
plt.title('Permutation Feature Importance (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()