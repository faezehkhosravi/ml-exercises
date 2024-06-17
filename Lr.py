import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

# Display basic information about the dataset
print(df.info())
print(df.describe())

# Fill missing values with the mean of each column (if any)
df.fillna(df.mean(), inplace=True)

# Create a new feature by multiplying two existing features
df['quality_ratio'] = df['fixed acidity'] / df['volatile acidity']

# Plot the distribution of the new feature
plt.figure(figsize=(10, 6))
sns.histplot(df['quality_ratio'], kde=True)
plt.title('Distribution of Quality Ratio')
plt.xlabel('Quality Ratio')
plt.ylabel('Frequency')
plt.show()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Prepare data for modeling
X = df[['fixed acidity', 'volatile acidity', 'quality_ratio']]
y = df['quality']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target values for the test set
y_pred = model.predict(X_test)

# Calculate the mean squared error of the predictions
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()
