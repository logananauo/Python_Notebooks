import pandas as pd

Q1 = df['purchase_amount'].quantile(0.25)

Q3 = df['purchase_amount'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR

upper = Q3 + 1.5 * IQR

# Filter or flag (flagging is often safer)

df['is_outlier'] = (

    (df['purchase_amount'] < lower) |

    (df['purchase_amount'] > upper)

).astype(int)
