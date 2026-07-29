### BOOSTING ###

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


### synthetic classification dataset
X, y = make_classification(n_samples=1000, 
                           n_features=5, 
                           n_informative=4, 
                           n_redundant=1, 
                           random_state=42) 


### train-test split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    stratify=y, 
                                                    test_size=0.3, 
                                                    random_state=42)


### define stump
stump = DecisionTreeClassifier(max_depth=1, random_state=42)


### define AdaBoostClassifier with Decision Tree as estimator
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=2, random_state=42), 
                         n_estimators=90, 
                         learning_rate=1.0, 
                         random_state=42)


### fit models and predict
stump.fit(X_train, y_train)
stump_preds = stump.predict(X_test)

ada.fit(X_train, y_train)
ada_preds = ada.predict(X_test)


### compute and display performance metrics
def compute_metrics(name, y_test, y_preds):
    accuracy  = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average='weighted')
    recall    = recall_score(y_test, y_preds, average='weighted')
    f1        = f1_score(y_test, y_preds, average='weighted')

    print(f'''{name} (test data):
    o accuracy:           {accuracy:.3f}
    o weighted precision: {precision:.3f}
    o weighted recall:    {recall:.3f}
    o weighted f1 score:  {f1:.3f}
    ''')
    return


compute_metrics('AdaBoost', y_test, ada_preds)
compute_metrics('Stump', y_test, stump_preds)