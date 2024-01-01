import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Sample data (you can replace this with your dataset)
data = {
    'Age': [25, 45, 30, 50, 35],
    'BMI': [22, 30, 25, 35, 28],
    'Children': [0, 2, 1, 3, 1],
    'Smoker': [0, 1, 0, 1, 0],  # 0: Non-smoker, 1: Smoker
    'Health_Costs': [5000, 10000, 7000, 15000, 8000],
}

df = pd.DataFrame(data)

# Separate features (X) and target variable (y)
X = df.drop('Health_Costs', axis=1)
y = df['Health_Costs']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Example: Predict health costs for a new individual
new_individual = np.array([35, 28, 1, 0]).reshape(1, -1)
predicted_health_costs = model.predict(new_individual)
print(f'Predicted Health Costs for the new individual: ${predicted_health_costs[0]:,.2f}')

# Visualization (optional)
plt.scatter(X_test['Age'], y_test, color='black', label='Actual')
plt.scatter(X_test['Age'], y_pred, color='blue', label='Predicted')
plt.xlabel('Age')
plt.ylabel('Health Costs')
plt.legend()
plt.show()
