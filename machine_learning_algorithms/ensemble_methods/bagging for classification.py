### BAGGING FOR CLASSIFICATION ###

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


### synthetic classification dataset
X, y = make_classification(n_samples=1000, # number of observations (rows)
                           n_features=14, # predictor variables per observation
                           n_informative=9, # Statistically significant predictor variables
                           n_redundant=5, # number of features that are correlated with informative
                           random_state=42) # reproduceability


### train-test split 70/30
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42)


### define base classifier
base_clf = DecisionTreeClassifier(random_state=0)


### define bagging classifier
bag_clf = BaggingClassifier(estimator=base_clf, 
                            n_estimators=10, 
                            bootstrap=True, 
                            random_state=42)


### fit both classifiers
base_clf.fit(X_train, y_train)
bag_clf.fit(X_train, y_train)


### predict on test set
single_preds = base_clf.predict(X_test)
bag_preds = bag_clf.predict(X_test)


### compare performance
accuracy  = accuracy_score(y_test, single_preds)
precision = precision_score(y_test, single_preds)
recall    = recall_score(y_test, single_preds)
f1        = f1_score(y_test, single_preds)
name      = 'Single Decision Tree'
print(f'{name} (test data) -- accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, F1: {f1:.3f}')

accuracy  = accuracy_score(y_test, bag_preds)
precision = precision_score(y_test, bag_preds)
recall    = recall_score(y_test, bag_preds)
f1        = f1_score(y_test, bag_preds)
name      = 'Bagging (10 Trees)'
print(f'{name} (test data) -- accuracy: {accuracy:.3f}, precision: {precision:.3f}, recall: {recall:.3f}, F1: {f1:.3f}')


### visualize with confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
cm_base = confusion_matrix(y_test, single_preds)
cm_disp_base = ConfusionMatrixDisplay(confusion_matrix=cm_base)
cm_disp_base.plot(ax=ax[0], cmap='Blues')
ax[0].set_title('Base Tree')

cm_bag = confusion_matrix(y_test, bag_preds)
cm_disp_bag = ConfusionMatrixDisplay(confusion_matrix=cm_bag)
cm_disp_bag.plot(ax=ax[1], cmap='Greens')
ax[1].set_title('Bagging (10 Trees)')

plt.tight_layout()
plt.show()
