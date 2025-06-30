# Matplotlib Basics: Explanations for matplotlibBasics.ipynb

This document explains each section and code block from `matplotlibBasics.ipynb`, which demonstrates basic data visualization techniques using matplotlib and the Iris dataset. Each section includes the actual Python code used, followed by an explanation.

---

## Step 0: Setup
```python
# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris = pd.read_csv(url, names=columns)

iris.head()
```
**Explanation:**
- Imports pandas and matplotlib for data handling and plotting.
- Loads the Iris dataset from the UCI repository into a DataFrame.
- Displays the first few rows to preview the data.

---

## Step 1: Line Plot
```python
# Line plot of sepal length
plt.plot(iris['sepal_length'])
plt.title('Line Plot of Sepal Length')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length')
plt.grid(True)
plt.show()
```
**Explanation:**
- Plots the `sepal_length` for all samples as a line plot.
- Adds a title, axis labels, and grid for clarity.
- `plt.show()` displays the plot.

---

## Step 2: Scatter Plot
```python
# Scatter plot between Sepal Length and Sepal Width
plt.scatter(iris['sepal_length'], iris['sepal_width'], c='green', alpha=0.6)
plt.title('Sepal Length vs Width')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()
```
**Explanation:**
- Creates a scatter plot to show the relationship between sepal length and sepal width.
- Sets color and transparency for better visualization.
- Adds title and axis labels.

---

## Step 3: Bar Plot
```python
# Bar plot of species count
species_counts = iris['species'].value_counts()

plt.bar(species_counts.index, species_counts.values, color=['red', 'blue', 'green'])
plt.title('Count of Each Iris Species')
plt.xlabel('Species')
plt.ylabel('Count')
plt.show()
```
**Explanation:**
- Counts the number of samples for each species using `value_counts()`.
- Plots these counts as a bar plot with different colors for each species.
- Adds title and axis labels.

---

## Step 4: Histogram
```python
# Histogram of Petal Length
plt.hist(iris['petal_length'], bins=20, color='purple', edgecolor='black')
plt.title('Histogram of Petal Length')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.show()
```
**Explanation:**
- Plots a histogram to show the distribution of petal length values.
- Uses 20 bins, sets color and edge color for clarity.
- Adds title and axis labels.

---

## Step 5: Box Plot
```python
# Box plot of all features
iris.drop(columns='species').plot(kind='box', figsize=(10, 6))
plt.title('Boxplot of Iris Features')
plt.grid(True)
plt.show()
```
**Explanation:**
- Drops the `species` column and creates box plots for all numerical features.
- Sets figure size, adds title, and grid.
- Box plots show the median, quartiles, and outliers for each feature.

---

## Step 6: Multiple Subplots
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sepal subplot
axes[0].scatter(iris['sepal_length'], iris['sepal_width'], color='teal')
axes[0].set_title('Sepal: Length vs Width')

# Petal subplot
axes[1].scatter(iris['petal_length'], iris['petal_width'], color='orange')
axes[1].set_title('Petal: Length vs Width')

plt.tight_layout()
plt.show()
```
**Explanation:**
- Creates a figure with two subplots side by side.
- First subplot: sepal length vs. sepal width.
- Second subplot: petal length vs. petal width.
- Sets titles for each subplot and uses `plt.tight_layout()` for spacing.

---

## Step 7: Customizing Style
```python
plt.style.use('ggplot')
plt.plot(iris['sepal_length'], label='Sepal Length')
plt.plot(iris['petal_length'], label='Petal Length')
plt.title('Comparison of Lengths')
plt.xlabel('Sample Index')
plt.ylabel('Length (cm)')
plt.legend()
plt.show()
```
**Explanation:**
- Applies the 'ggplot' style for improved aesthetics.
- Plots both sepal and petal lengths on the same plot with labels and legend.
- Adds title, axis labels, and legend for clarity.

---

Each section in the notebook demonstrates a key matplotlib concept or workflow, using the Iris dataset as a practical example. This covers the basics of line, scatter, bar, histogram, box plots, subplots, and style customization in matplotlib. 