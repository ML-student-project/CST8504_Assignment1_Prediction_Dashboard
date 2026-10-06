import data_processor as dp
import pandas as pd

df = pd.read_csv('../data/raw/framingham_heart_study.csv')

# Display the first 5 rows
print(df.head())

print('Number of rows:', df.shape[0])
print('\tNumber of columns:', df.shape[1])
print(df.columns)

pd.set_option('display.max_columns', None)
print(df.describe(include='all'))

# Count NaN values in each column
nan_counts = df.isna().sum()
print(nan_counts)
print(type(nan_counts))


dp.plot_data(df)
