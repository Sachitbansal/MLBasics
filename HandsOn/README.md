# Line Fitting Example: Step-by-Step Explanation

This section explains the code used for fitting a straight line to synthetic data using linear regression in `lineFittingKNN.ipynb`.

---

## Code & Explanation

```python
# Generate synthetic linear data
X, y = make_regression(n_samples=100, n_features=1, noise=15, random_state=42)
```
**Explanation:**
- `make_regression`: Function from `sklearn.datasets` to generate synthetic regression data.
  - `n_samples=100`: Number of data points (rows).
  - `n_features=1`: Number of input features (columns).
  - `noise=15`: Standard deviation of Gaussian noise added to the output (adds randomness).
  - `random_state=42`: Seed for reproducibility (same data every run).
- **Returns:**
  - `X`: 2D array of shape (100, 1), input features.
  - `y`: 1D array of shape (100,), target values.
- **Use:** Simulates real-world noisy linear data for regression.

---

```python
# Fit linear regression
model = LinearRegression()
model.fit(X, y)
```
**Explanation:**
- `LinearRegression()`: Creates a linear regression model object from `sklearn.linear_model`.
- `model.fit(X, y)`: Trains the model to find the best-fit line for the data.
  - `X`: Input features.
  - `y`: Target values.
- **Returns:** The model is fitted (internal parameters are set).
- **Use:** Learns the relationship between `X` and `y`.

---

```python
# Predict
y_pred = model.predict(X)
```
**Explanation:**
- `model.predict(X)`: Uses the trained model to predict target values for `X`.
- **Returns:**
  - `y_pred`: Predicted values (best-fit line values for each `X`).
- **Use:** To visualize and evaluate the model's fit.

---

```python
# Print coefficients
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
```
**Explanation:**
- `model.coef_`: Array of learned slope(s) (for each feature). Here, just one value.
- `model.intercept_`: Learned intercept (y-axis crossing point).
- **Use:** Shows the equation of the fitted line: `y = mX + c`.

---

## Summary
- **Purpose:** Demonstrates how to generate data, fit a regression line, visualize, and interpret the model.
- **Key Functions:** `make_regression`, `LinearRegression.fit`, `LinearRegression.predict`.
- **Visualization:** Helps understand model performance and the concept of linear regression.

# Regression on Real Data: California Housing Example

This section explains the code for fitting a linear regression model to the California Housing dataset using a single feature (Median Income).

---

```python
# Load dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target
```
**Explanation:**
- `fetch_california_housing()`: Loads the California housing dataset from `sklearn.datasets`.
  - **Returns:** A dictionary-like object with data and metadata.
- `pd.DataFrame(data.data, columns=data.feature_names)`: Converts the data to a pandas DataFrame with column names.
- `df['Target'] = data.target`: Adds the target variable (house value) as a new column.
- **Use:** Prepares the dataset for analysis and modeling.

---

```python
# Select a subset (for simplicity)
X = df[['MedInc']]  # Median income
y = df['Target']    # House value
```
**Explanation:**
- `X = df[['MedInc']]`: Selects the 'MedInc' (median income) column as the input feature (must be 2D for sklearn).
- `y = df['Target']`: Selects the target variable (house value).
- **Use:** Focuses on a single feature for simple linear regression.

---

```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
**Explanation:**
- `train_test_split`: Splits data into training and testing sets.
  - `X`, `y`: Features and target.
  - `test_size=0.2`: 20% of data for testing, 80% for training.
  - `random_state=42`: Ensures reproducibility.
- **Returns:**
  - `X_train`, `X_test`: Training and testing features.
  - `y_train`, `y_test`: Training and testing targets.
- **Use:** Evaluates model performance on unseen data.

---

```python
# Fit model
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
```
**Explanation:**
- `LinearRegression()`: Creates a linear regression model object.
- `reg_model.fit(X_train, y_train)`: Trains the model on the training data.
- **Returns:** The model is fitted (parameters learned).
- **Use:** Learns the relationship between median income and house value.

---

```python
# Predict
y_pred = reg_model.predict(X_test)
```
**Explanation:**
- `reg_model.predict(X_test)`: Predicts house values for the test set using the trained model.
- **Returns:**
  - `y_pred`: Predicted house values for the test set.
- **Use:** To evaluate and visualize model predictions.

---

## Model Evaluation: Metrics Explained

After making predictions, we evaluate the model using two key metrics:

```python
# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
```

### 1. Mean Squared Error (MSE)
- **Meaning:**
  - Measures the average squared difference between actual (`y_test`) and predicted (`y_pred`) values.
  - Lower values indicate better model performance (closer predictions).

- **Importance:**
  - Penalizes larger errors more than smaller ones (due to squaring).
  - Commonly used for regression tasks.
- **Inference:**
  - A lower MSE means the model's predictions are closer to the actual values.

# K-Nearest Neighbors (KNN) Classification: Iris Dataset Example

## What is K-Nearest Neighbors (KNN)?

K-Nearest Neighbors (KNN) is a simple, non-parametric algorithm used for classification and regression. It predicts the label of a new data point by looking at the 'k' closest points in the training data and choosing the most common class (for classification) or averaging their values (for regression). KNN is easy to use and works well for small datasets with non-linear boundaries.

---

```python
# Load dataset
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
```
**Explanation:**
- `load_iris()`: Loads the Iris flower dataset from `sklearn.datasets`.
  - **Returns:** A dictionary-like object with data and metadata.
- `pd.DataFrame(iris.data, columns=iris.feature_names)`: Converts the data to a pandas DataFrame with column names.
- `df['species'] = iris.target`: Adds the target variable (species) as a new column.
- **Use:** Prepares the dataset for classification.

---

```python
# Features and target
X = df.iloc[:, :4]
y = df['species']
```
**Explanation:**
- `X = df.iloc[:, :4]`: Selects the first four columns (features: sepal length, sepal width, petal length, petal width).
- `y = df['species']`: Selects the target variable (species).
- **Use:** Defines features and labels for classification.

---

```python
# Scale features (important for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```
**Explanation:**
- `StandardScaler()`: Creates a scaler object to standardize features (mean=0, std=1).
- `scaler.fit_transform(X)`: Fits the scaler to `X` and transforms it.
  - **Returns:** `X_scaled`, the standardized features.
- **Use:** KNN is distance-based, so feature scaling is crucial for fair comparison.

---

```python
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
```
**Explanation:**
- `train_test_split`: Splits data into training and testing sets.
  - `X_scaled`, `y`: Features and target.
  - `test_size=0.25`: 25% of data for testing, 75% for training.
  - `random_state=42`: Ensures reproducibility.
- **Returns:**
  - `X_train`, `X_test`: Training and testing features.
  - `y_train`, `y_test`: Training and testing targets.
- **Use:** Evaluates model performance on unseen data.

---

```python
# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
```
**Explanation:**
- `KNeighborsClassifier(n_neighbors=5)`: Creates a KNN classifier with 5 neighbors.
- `knn.fit(X_train, y_train)`: Trains the classifier on the training data.
- **Returns:** The model is fitted (stores training data for neighbor lookup).
- **Use:** Learns to classify based on the majority class among the 5 nearest neighbors.

---

```python
# Predict
y_pred = knn.predict(X_test)
```
**Explanation:**
- `knn.predict(X_test)`: Predicts the species for the test set using the trained KNN model.
- **Returns:**
  - `y_pred`: Predicted species labels for the test set.
- **Use:** To evaluate and analyze model predictions.

---

```python
# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
```
**Explanation:**
- `accuracy_score(y_test, y_pred)`: Computes the proportion of correct predictions.
- `classification_report(y_test, y_pred)`: Shows precision, recall, f1-score, and support for each class.
- `confusion_matrix(y_test, y_pred)`: Displays a matrix of actual vs. predicted labels.
- **Use:** Measures and summarizes the performance of the classifier.

**Classification Report Metrics:**
- **Precision:** Correct positive predictions / all predicted positives
- **Recall:** Correct positive predictions / all actual positives
- **F1-score:** Harmonic mean of precision and recall

**How to Read the Confusion Matrix:**
- The confusion matrix shows how many predictions were correct or incorrect for each class.
- **Rows** represent the actual classes, **columns** represent the predicted classes.
- The diagonal elements (top-left to bottom-right) show correct predictions; off-diagonal elements show misclassifications.

**Example:**
Suppose the confusion matrix for a 3-class problem is:

|     | Pred 0 | Pred 1 | Pred 2 |
|-----|--------|--------|--------|
| **Actual 0** |   5    |   0    |   0    |
| **Actual 1** |   1    |   4    |   0    |
| **Actual 2** |   0    |   0    |   6    |

- Here, 5 samples of class 0 were correctly predicted as 0, 4 of class 1 as 1, and 6 of class 2 as 2.
- 1 sample of class 1 was incorrectly predicted as class 0.
- The higher the diagonal values, the better the model's performance.

---
