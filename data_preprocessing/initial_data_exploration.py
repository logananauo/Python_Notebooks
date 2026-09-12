import pandas as pd

df = pd.read_csv('')

print(df.shape)          # how many rows and columns?
print(df.dtypes)         # what data types are we dealing with?
print(df.isnull().sum()) # where are the gaps?
print(df.describe())     # summary statistics

### check for duplicates
print(f'Duplicate rows: {df.duplicated().sum()}')
