### RANDOM FOREST FOR CLASSIFICATIONS ###

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay


### synthetic classification dataset (3 classes, 5 features)
X, y = make_classification(n_samples=300, 
                           n_features=5, 
                           n_informative=3, 
                           n_classes=3, 
                           #weights=[0.7, 0.2, 0.1], # these weights correspond to an imbalanced class distribution
                           weights=[0.33, 0.33, 0.34], # these weights correspond to a balanced class distribution
                           random_state=42)


### train-test split 70-30
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.3, 
                                                    random_state=42)


### train a single decision tree vs. a random forest
tree = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

tree.fit(X_train, y_train)
rf.fit(X_train, y_train)


### generate predictions
tree_preds = tree.predict(X_test)
rf_preds   = rf.predict(X_test)


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


compute_metrics('Single Decision Tree', y_test, tree_preds)
compute_metrics('Random Forest', y_test, rf_preds)


### confusion matrices
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

cm_base = confusion_matrix(y_test, tree_preds)
cm_disp_base = ConfusionMatrixDisplay(confusion_matrix=cm_base)
cm_disp_base.plot(ax=ax[0], cmap='Blues')
ax[0].set_title('Single Tree')

cm_bag = confusion_matrix(y_test, rf_preds)
cm_disp_bag = ConfusionMatrixDisplay(confusion_matrix=cm_bag)
cm_disp_bag.plot(ax=ax[1], cmap='Greens')
ax[1].set_title('Random Forest')

plt.tight_layout()
plt.show()
