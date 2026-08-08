
#!pip install lime

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


### generate synthetic binary classification dataset
X, y = make_classification(n_samples=1000, 
                           n_features=5, 
                           n_informative=3, 
                           n_redundant=0, 
                           random_state=42, 
                           shuffle=False)


### to dataframe
feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)
y_series = pd.Series(y, name='Target')
X_df.head()




# -------------------------------- #
### Train a Black-Box Classifier ###
# -------------------------------- #

### train-test split
X_train, X_test, y_train, y_test = train_test_split(X_df, 
                                                    y_series, 
                                                    test_size=0.2, 
                                                    random_state=42)


### train model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

print('Model accuracy on test set:', rf.score(X_test, y_test))


### Create LIME explainer
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(training_data=np.array(X_train), 
                                 feature_names=feature_names, 
                                 class_names=['Class 0', 'Class 1'], 
                                 mode='classification', 
                                 discretize_continuous=True)

def predict_fn_with_names(x):
    return rf.predict_proba(pd.DataFrame(x, columns=feature_names))


### choose an instance to explain
i = 5
instance = X_test.iloc[i]


### generate explanation
exp = explainer.explain_instance(data_row=instance, 
                                 predict_fn=predict_fn_with_names, 
                                 num_features=5)


### show explanation in notebook
exp.show_in_notebook(show_table=True)


### text format of the explanation
print(exp.as_list())


### convert explanation to DataFrame
exp_df = pd.DataFrame(exp.as_list(), columns=['Feature', 'Effect on Prediction'])
print(exp_df)


