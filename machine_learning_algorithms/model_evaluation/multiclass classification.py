### MULTICLASS CLASSIFICATION ###

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


### iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names


### train-test split 70/30
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    stratify=y,
                                                    test_size=0.3, 
                                                    random_state=42)


### train decision tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


### confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


### compute TP/FP/FN/TN for each class
metrics = []
for i in range(len(class_names)):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP
    FN = cm[i, :].sum() - TP
    TN = cm.sum() - (TP + FP + FN)
    metrics.append({'Class': class_names[i], 'TP':TP, 'FP':FP, 'FN':FN, 'TN':TN})

metrics_df = pd.DataFrame(metrics)
metrics_df


### generate the classification report
report = classification_report(y_test, y_pred, labels=[0, 1, 2])
print(report)
for i in range(3):
    print(f'i={i}, label={class_names[i]}')


