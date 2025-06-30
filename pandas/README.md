# Pandas Basics: Explanations for pandasBasics.ipynb

This document explains each section and code block from `pandasBasics.ipynb`, which demonstrates pandas fundamentals using the Iris dataset.

---

## 📥 Download or Load Iris Dataset (CSV)
- The Iris dataset is loaded directly from the UCI Machine Learning Repository using `pd.read_csv()`.
- Column names are specified for clarity.

```python
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, names=columns)
```

---

## 1. Core Data Structures
- The main pandas data structure is the **DataFrame** (table of data).
- The code checks the type, data types of columns, and lists all column names.

```python
print(type(iris))             # DataFrame type
print(iris.dtypes)            # Data types of each column
print(iris.columns.tolist())  # List of columns
```

---

## 2. Reading & Writing Data
- Demonstrates saving a DataFrame to a CSV file and reading it back.
- `to_csv()` writes the DataFrame to disk; `read_csv()` loads it.

```python
iris.to_csv("iris_copy.csv", index=False)         # Write to CSV
iris_loaded = pd.read_csv("iris_copy.csv")        # Read from CSV
```

---

## 3. Data Selection & Indexing
- Shows how to select columns, multiple columns, and use both index-based (`iloc`) and label-based (`loc`) slicing.
- Useful for extracting specific data for analysis.

```python
iris['sepal_length'].head()                # Single column
iris[['sepal_length', 'species']].head()   # Multiple columns
iris.iloc[0:5, 0:2]                        # Index-based slicing
iris.loc[0:4, ['sepal_width', 'species']]  # Label-based slicing
```

---

## 4. Data Cleaning
- Introduces a missing value (NaN) and demonstrates how to:
  - Count missing values with `isnull().sum()`
  - Fill missing values with `fillna()`
  - Drop rows with missing values using `dropna()`
- Also shows how to rename columns and change data types.

```python
iris.loc[0, 'sepal_length'] = None                 # Introduce NaN
iris.isnull().sum()                                # Count NaNs
iris_filled = iris.fillna(0)                       # Fill NaN with 0
iris_dropped = iris.dropna()                       # Drop NaN rows
iris = iris.rename(columns={'sepal_length': 'sepal_len'})  # Rename
iris['petal_width'] = iris['petal_width'].astype(float)    # Change dtype
```

---

## 5. Data Manipulation
- Adds a new column as a sum of two columns.
- Drops a column.
- Performs string operations (convert to uppercase).
- Uses `apply()` with a lambda to create a new column with squared values.

```python
iris['sepal_plus_petal'] = iris['sepal_len'] + iris['petal_length']   # Add new column
iris = iris.drop(columns=['sepal_plus_petal'])                        # Drop column
iris['species_upper'] = iris['species'].str.upper()                   # String operation
iris['squared_len'] = iris['sepal_len'].apply(lambda x: x**2 if pd.notnull(x) else x)  # Lambda
```

---

## 6. Aggregation & Grouping
- Groups data by species and calculates:
  - Mean of sepal length
  - Mean and max of petal length
- Uses `value_counts()` to count the number of samples per species.

```python
grouped = iris.groupby('species').agg({
    'sepal_len': 'mean',
    'petal_length': ['mean', 'max']
})
print(grouped)
print(iris['species'].value_counts())
```

---

## 7. Sorting & Ranking
- Sorts the DataFrame by sepal length in descending order.
- Adds a new column with the rank of each sepal length value.

```python
iris_sorted = iris.sort_values(by='sepal_len', ascending=False)
iris['sepal_rank'] = iris['sepal_len'].rank()
```

---

## 8. Merging & Joining
- Creates a new DataFrame with species labels.
- Merges the label DataFrame with the main DataFrame using `merge()`.
- This is useful for adding categorical or numerical labels to your data.

```python
label_df = pd.DataFrame({
    'species': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    'label': [0, 1, 2]
})
merged_df = pd.merge(iris, label_df, on='species', how='left')
```

---

Each section in the notebook demonstrates a key pandas concept or workflow, using the Iris dataset as a practical example. This covers the basics of loading, exploring, cleaning, manipulating, aggregating, sorting, and merging data with pandas.
