import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Core Data Structures
# Series
s1 = pd.Series([1, 2, 3, np.nan, 5])
print('Series s1:\n', s1)

# DataFrame from dict
df_dict = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print('\nDataFrame from dict:\n', df_dict)

# DataFrame from list of lists
lst = [[1, 'a'], [2, 'b'], [3, 'c']]
df_list = pd.DataFrame(lst, columns=['num', 'char'])
print('\nDataFrame from list of lists:\n', df_list)

# DataFrame from NumPy array
arr = np.arange(6).reshape(2, 3)
df_np = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print('\nDataFrame from NumPy array:\n', df_np)

# DataFrame from CSV (example, not executed)
# df_csv = pd.read_csv('example.csv')

# 2. Reading & Writing Data
# Reading (examples, not executed)
# df_csv = pd.read_csv('file.csv')
# df_excel = pd.read_excel('file.xlsx')
# df_json = pd.read_json('file.json')

# Writing (examples, not executed)
# df_dict.to_csv('out.csv')
# df_dict.to_excel('out.xlsx')

# 3. Data Selection & Indexing
print('\nSelecting column A:', df_dict['A'])
print('Selecting first row with .loc:', df_dict.loc[0])
print('Selecting first row with .iloc:', df_dict.iloc[0])
print('Selecting with .at:', df_dict.at[0, 'A'])
print('Selecting with .iat:', df_dict.iat[0, 0])

# Boolean indexing
print('\nRows where A > 1:\n', df_dict[df_dict['A'] > 1])

# Slicing rows/columns
print('First two rows:\n', df_dict[:2])
print('Column B as DataFrame:\n', df_dict[['B']])

# Setting/Resetting index
indexed = df_dict.set_index('A')
print('\nSet index to column A:\n', indexed)
reset = indexed.reset_index()
print('Reset index:\n', reset)

# 4. Data Cleaning
nan_df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
print('\nDataFrame with NaNs:\n', nan_df)
print('isnull():\n', nan_df.isnull())
print('dropna():\n', nan_df.dropna())
print('fillna(0):\n', nan_df.fillna(0))

# Renaming columns
renamed = df_dict.rename(columns={'A': 'a_col', 'B': 'b_col'})
print('\nRenamed columns:\n', renamed)

# Changing dtypes
print('A as float:\n', df_dict['A'].astype(float))

# 5. Data Manipulation
# Filtering rows
filtered = df_dict[df_dict['A'] > 1]
print('\nFiltered rows (A > 1):\n', filtered)

# Adding/removing columns
df_dict['C'] = ['x', 'y', 'z']
print('Added column C:\n', df_dict)
df_removed = df_dict.drop('C', axis=1)
print('Removed column C:\n', df_removed)

# String operations
str_df = pd.DataFrame({'text': ['Hello', 'world', 'Pandas', 'Numpy']})
print('Lowercase text:\n', str_df['text'].str.lower())
print('Contains "an":\n', str_df['text'].str.contains('an'))

# Apply functions
print('Apply len to text:\n', str_df['text'].apply(len))
map_df = pd.DataFrame({'num': [1, 2, 3]})
print('Map square to num:\n', map_df['num'].map(lambda x: x**2))

# 6. Aggregation & Grouping
group_df = pd.DataFrame({'Category': ['A', 'A', 'B'], 'Value': [10, 15, 10]})
grouped = group_df.groupby('Category').agg({'Value': ['sum', 'mean']})
print('\nGrouped by Category:\n', grouped)
print('Value counts:\n', group_df['Category'].value_counts())

# 7. Sorting & Ranking
sorted_df = group_df.sort_values('Value', ascending=False)
print('\nSorted by Value descending:\n', sorted_df)
print('Sort by index:\n', group_df.sort_index())

# 8. Merging & Joining
df1 = pd.DataFrame({'key': ['A', 'B'], 'val1': [1, 2]})
df2 = pd.DataFrame({'key': ['A', 'B'], 'val2': [3, 4]})
merged = pd.merge(df1, df2, on='key')
print('\nMerged DataFrames:\n', merged)
joined = df1.join(df2.set_index('key'), on='key', rsuffix='_r')
print('Joined DataFrames:\n', joined)
concat_df = pd.concat([df1, df2], axis=0)
print('Concatenated DataFrames (rows):\n', concat_df)
# append is deprecated, use concat

# 9. Basic Visualization
# (This will display a plot if run in a notebook or script with GUI)
plot_df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
plot_df.plot(x='x', y='y', kind='line', title='Line Plot')
plt.savefig('plot.png')  # Save plot as file
plt.close()
print('\nPlot saved as plot.png')

# 10. Descriptive Statistics
print('\nDescriptive statistics:\n', plot_df.describe())
print('Info:')
plot_df.info()
print('Shape:', plot_df.shape)
print('Head:\n', plot_df.head())
print('Tail:\n', plot_df.tail())
