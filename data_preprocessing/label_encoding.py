import pandas as pd
from sklearn.preprocessing import LabelEncoder

### one-hot encoding (most common)
df = pd.get_dummies(df, columns=['region', 'product_type'])

### label encoding for ordinal features
le = LabelEncoder()
df['tier_encoded'] = le.fit_transform(df['tier'])
