import pandas as pd
from sklearn.impute import SimpleImputer

### median imputation (robust to outliers)
imputer = SimpleImputer(strategy='median') # change to 'mean' for mean imputation

df[['age', 'income']] = imputer.fit_transform(df[['age', 'income']])

### create missingness indicator before imputing
df['income_missing'] = df['income'].isnull().astype(int)

### forward-fill for time series data
df['daily_revenue'] = df['daily_revenue'].ffill()
