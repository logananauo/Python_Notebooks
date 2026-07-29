### VOTING FOR CLASSIFICATION ###

from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


### synthetic classification dataset
X, y = make_classification(n_samples=500,
                           n_features=6,
                           n_informative=5,
                           n_redundant=1,
                           n_repeated=0,
                           n_classes=2, 
                           random_state=42)


### for pandas dataframes
#X = df[['X1', 'X2']]
#y = df['y']


### train-test split 75/25
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    stratify=y, 
                                                    test_size=0.25, 
                                                    random_state=42)


### base learners
clf1 = KNeighborsClassifier(n_neighbors=3)
clf2 = KNeighborsClassifier(n_neighbors=4)
clf3 = KNeighborsClassifier(n_neighbors=5)
clf4 = DecisionTreeClassifier(max_depth=3, random_state=42)
clf5 = DecisionTreeClassifier(max_depth=4, random_state=42)


### create a hard voting ensemble of these classifiers
voting_clf = VotingClassifier(estimators=[
    ('knn-3', clf1), ('knn-4', clf2), ('knn-5', clf3), ('dt_3', clf4), ('dt_4', clf5)
], voting='hard')


### train individual models and the ensemble
for clf in (clf1, clf2, clf3, clf4, clf5, voting_clf):
    clf.fit(X_train, y_train)


### evaulate quality on the test set
for name, clf in [('KNN-3', clf1), ('KNN-4', clf2), ('KNN-5', clf3), ('DecisionTree_3', clf4), ('DecisionTree_4', clf5), ('Ensemble', voting_clf)]:
    y_pred = clf.predict(X_test)
    accuracy       = accuracy_score(y_test, y_pred)
    precision      = precision_score(y_test, y_pred)
    recall         = recall_score(y_test, y_pred)
    f1             = f1_score(y_test, y_pred)
    
    print(f'{name:12} test -- accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, F1: {f1:.3f}')
