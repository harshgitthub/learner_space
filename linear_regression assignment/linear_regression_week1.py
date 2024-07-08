import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Aim is to predict the marks of students of the test data
# Use the file named 'Training data.xlsx' to train the model

# Load training data
data = pd.read_excel('Training data.xlsx')
x_train = data.iloc[:, 0:8].to_numpy()
y_train = data.iloc[:, 8].to_numpy().reshape(-1, 1)

# # Plot y_train with different features to get an idea whether to add some features or not
# for i in range(x_train.shape[1]):
#     plt.figure()
#     plt.scatter(x_train[:, i], y_train)
#     plt.xlabel(f'Feature {i}')
#     plt.ylabel('Marks')
#     plt.title(f'Feature {i} vs Marks')
#     plt.show()

# Feature encoding for non-numeric columns
def feature_changing(x_train):
    for i in range(x_train.shape[1]):
        # Identify columns with non-numeric data
        if not np.issubdtype(x_train[:, i].dtype, np.number):
            # Unique values in the column
            unique_values = np.unique(x_train[:, i])
            # Create a mapping from unique values to integers
            mapping = {value: idx for idx, value in enumerate(unique_values)}
            # Apply the mapping
            x_train[:, i] = np.vectorize(mapping.get)(x_train[:, i])
    return x_train

x_train = feature_changing(x_train)

# Ensure x_train is of numeric type
x_train = x_train.astype(float)

# Standardization (Z-score normalization)
def z_score(x_train):
    x_std = np.std(x_train, axis=0)
    x_mean = np.mean(x_train, axis=0)
    x_train = (x_train - x_mean) / x_std
    return x_train, x_std, x_mean

x_train, x_std, x_mean = z_score(x_train)

# Cost function (Mean Squared Error)
def cost(x_train, y_train, w, b):
    m = len(x_train)
    total_cost = 0
    for i in range(m):
        total_cost += ((np.dot(x_train[i], w) + b) - y_train[i]) ** 2
    loss = (1 / (2 * m)) * total_cost
    return loss

# Gradient Descent
def gradient_descent(x_train, y_train, w, b, learning_rate=0.03, epochs=50):
    m = len(x_train)
   
    rw = np.zeros_like(w)
    rb = 0
    for i in range(m):
        error = (np.dot(x_train[i], w) + b) - y_train[i]
        rw += error * x_train[i].reshape(-1, 1)
        rb += error
    rw /= m
    rb /= m
    for epoch in range(epochs):
        w -= learning_rate * rw  # Update weights
        b -= learning_rate * rb  # Update bias
    return w, b



# Initialize weights and bias
np.random.seed(2147483647)
w = np.random.randn(x_train.shape[1], 1)
b = np.random.randn(1)

# Training the model
old_cost = float('inf')
while True:
    current_cost = cost(x_train, y_train, w, b)
    if abs(old_cost - current_cost) < 0.00001:
        break
    old_cost = current_cost
    w, b = gradient_descent(x_train, y_train, w, b)

# Load and preprocess test data
x_predict = pd.read_excel('Test data.xlsx').iloc[:, :8].to_numpy()
x_predict = feature_changing(x_predict)
x_predict = x_predict.astype(float)
x_predict = (x_predict - x_mean) / x_std
ans = pd.read_excel('Test data.xlsx').iloc[:, 8].to_numpy()

# Predicting the test data
y_predict = np.dot(x_predict, w) + b

# Calculating accuracy
accuracy = np.mean(np.abs(y_predict.flatten() - ans) < 0.5) * 100
ok = 'Congratulations' if accuracy > 95 else 'Optimization required'
print(f"{ok}, your accuracy is {accuracy}%")
