### STACKING ###

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


### synthetic classification dataset
X, y = make_classification(n_samples=1000,
                           n_features=6, # 6 predictor variables
                           n_informative=4, # 4 useful features
                           n_redundant=1, # 1 correlated feature
                           n_repeated=0,
                           n_classes=2, # Binary classification
                           random_state=42)


### for pandas dataframes
#X = df[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
#y = df['y']


### train-test split 70/30
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    stratify=y, 
                                                    test_size=0.3, 
                                                    random_state=42)


### define base models
base_models = [
    ('Logistic Regression', LogisticRegression(max_iter=1000)), 
    ('Decision Tree (depth=3)', DecisionTreeClassifier(max_depth=3, random_state=42)), 
    ('Decision Tree (depth=5)', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('KNN (k=5)', KNeighborsClassifier(n_neighbors=5)), 
    ('KNN (k=7)', KNeighborsClassifier(n_neighbors=7))
]


### define stacking ensemble with logistic regression meta-model
stack_clf = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)


### fit stacking model to training data
stack_clf.fit(X_train, y_train)
stack_preds = stack_clf.predict(X_test)


### define function for computing performance metrics
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


### compute the metrics
compute_metrics("Stacked", y_test, stack_preds)


### evaluate each base model and the stacked model
for name, model in base_models:
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    compute_metrics(name, y_test, y_preds)
    