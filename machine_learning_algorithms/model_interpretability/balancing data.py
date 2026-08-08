#!pip install imbalanced-learn

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler


### generate imbalanced classification dataset
X, y = make_classification(n_classes=2, 
                           weights=[0.93, 0.08], 
                           n_informative=3, 
                           n_redundant=0, 
                           flip_y=0, 
                           n_features=5, 
                           n_clusters_per_class=1, 
                           n_samples=1000, 
                           random_state=42)


### check class distribution
unique, counts = np.unique(y, return_counts=True)
print('Original class distribution:', dict(zip(unique, counts)))


### train-test split
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    stratify=y, 
                                                    random_state=42)


### train a classifier on imbalanced data
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('\n======= Classification Report (Before Oversampling): =======')
print(classification_report(y_test, y_pred, digits=3))


### apply random oversampling
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)


### check new class distribution
unique_res, counts_res = np.unique(y_resampled, return_counts=True)
print('\nResampled class distribution:', dict(zip(unique_res, counts_res)))


### Retrain model on balanced data
model_resampled = RandomForestClassifier(random_state=42)
model_resampled.fit(X_resampled, y_resampled)
y_pred_resampled = model_resampled.predict(X_test)
print('\n======= Classification Report (After Oversampling): =======')
print(classification_report(y_test, y_pred_resampled, digits=3))