# NumPy Basics: Code Explanation

This document explains the basic usage of the NumPy library as demonstrated in `basics.py`. NumPy is a fundamental package for scientific computing in Python, providing support for arrays, mathematical functions, and more.

## 1. Creating NumPy Arrays
```python
arr1 = np.array([1, 2, 3, 4, 5])  # 1D array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array
```
- `np.array` creates a NumPy array from a Python list or list of lists.
- `arr1` is a one-dimensional array; `arr2` is a two-dimensional array (matrix).

## 2. Array Attributes
```python
shape = arr2.shape  # Shape of the array
ndim = arr2.ndim    # Number of dimensions
dtype = arr1.dtype  # Data type of elements
```
- `.shape` returns the dimensions of the array (rows, columns).
- `.ndim` gives the number of dimensions (axes).
- `.dtype` shows the data type of the array elements.

## 3. Creating Arrays with Built-in Functions
```python
zeros = np.zeros((2, 3))      # 2x3 array of zeros
ones = np.ones((2, 3))        # 2x3 array of ones
arange = np.arange(0, 10, 2)  # Array with values from 0 to 8 with step 2
linspace = np.linspace(0, 1, 5)  # 5 values from 0 to 1 (inclusive)
```
- `np.zeros` and `np.ones` create arrays filled with zeros or ones.
- `np.arange(start, stop, step)` creates an array with regularly incrementing values.
- `np.linspace(start, stop, num)` creates an array of evenly spaced values between start and stop.

## 4. Basic Operations
```python
add = arr1 + 10           # Add 10 to each element
multiply = arr1 * 2       # Multiply each element by 2
sum_all = arr1.sum()      # Sum of all elements
mean_all = arr1.mean()    # Mean of all elements
max_val = arr1.max()      # Maximum value
min_val = arr1.min()      # Minimum value
```
- Arithmetic operations are element-wise.
- `.sum()`, `.mean()`, `.max()`, `.min()` are aggregation functions.

## 5. Indexing and Slicing
```python
first_element = arr1[0]         # First element
last_two = arr1[-2:]            # Last two elements
row_1 = arr2[0]                 # First row of 2D array
col_2 = arr2[:, 1]              # Second column of 2D array
subarray = arr2[0:2, 1:3]       # Subarray from arr2
```
- Indexing retrieves specific elements or slices.
- `arr2[:, 1]` selects all rows, second column.
- `arr2[0:2, 1:3]` selects a submatrix.

## 6. Reshaping and Flattening
```python
reshaped = arr1.reshape((5, 1)) # Reshape to 5x1 array
flattened = arr2.flatten()      # Flatten 2D array to 1D
```
- `.reshape` changes the shape of an array.
- `.flatten` converts a multi-dimensional array to 1D.

## 7. Mathematical Functions
```python
sqrt = np.sqrt(arr1)            # Square root
exp = np.exp(arr1)              # Exponential
sin = np.sin(arr1)              # Sine
```
- NumPy provides vectorized mathematical functions that operate element-wise.

## 8. Aggregation Along Axes
```python
sum_axis0 = arr2.sum(axis=0)    # Sum along columns
sum_axis1 = arr2.sum(axis=1)    # Sum along rows
```
- The `axis` parameter specifies the dimension for aggregation: `axis=0` for columns, `axis=1` for rows.

## 9. Copying Arrays
```python
arr1_copy = arr1.copy()         # Create a copy of arr1
```
- `.copy()` creates a new array with the same data, independent of the original.

---

These examples cover the essential basics of NumPy for array creation, manipulation, and computation. NumPy is highly efficient and forms the foundation for most scientific and machine learning libraries in Python.
