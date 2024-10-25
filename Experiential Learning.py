from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
#from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the uploaded CSV file to inspect its contents
file_path = "C:\\Users\\Tenzin kunga\\Downloads\\ASCP\\abroad  - Sheet1.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the dataset and summary information
data.head(), data.info()

# Convert the 'FEES' column to a numeric data type and handle any errors (e.g., remove commas if present)
data['FEES'] = pd.to_numeric(data['FEES'], errors='coerce')

# Check for missing values and the structure of the dataset after conversion
missing_values = data.isnull().sum()
print(missing_values)

# Drop row with missing FEES value
data = data.dropna(subset=['FEES'])

# One-hot encode 'COUNTRY' and 'COURSE TYPE' columns
data_encoded = pd.get_dummies(data[['COUNTRY', 'COURSE TYPE', 'FEES']], columns=['COUNTRY', 'COURSE TYPE'], drop_first=True)

# Define features and target variable
X = data_encoded.drop('FEES', axis=1)
y = data_encoded['FEES']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)

# Visualization 1: Distribution of Fees by Country and Course Type
plt.figure(figsize=(12, 6))
sns.barplot(data=data, x='COUNTRY', y='FEES', hue='COURSE TYPE', errorbar=None)
plt.title('Average Fees by Country and Course Type')
plt.ylabel('Fees')
plt.xticks(rotation=45)
plt.legend(title='Course Type')
plt.tight_layout()
plt.show()

# Visualization 2: Predicted vs. Actual Fees
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Fees')
plt.ylabel('Predicted Fees')
plt.title('Predicted vs. Actual Fees')
plt.tight_layout()
plt.show()
