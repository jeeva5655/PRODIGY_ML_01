import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# Step 1: Generate a new synthetic dataset with different values
np.random.seed(100)  # Using a different seed for different values
num_samples = 100

square_footage = np.random.randint(600, 4000, num_samples)
bedrooms = np.random.randint(1, 7, num_samples)
bathrooms = np.random.randint(1, 5, num_samples)
prices = (square_footage * 350) + (bedrooms * 12000) + (bathrooms * 6000) + np.random.randint(-25000, 25000, num_samples)

data = pd.DataFrame({
    'SquareFootage': square_footage,
    'Bedrooms': bedrooms,
    'Bathrooms': bathrooms,
    'Price': prices
})

# Step 2: Explore and preprocess the data
print(data.head())

# Step 3: Split the data into training and testing sets
X = data[['SquareFootage', 'Bedrooms', 'Bathrooms']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_s

# Step 4: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model's performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Model Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)

import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plots of features vs. price
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
sns.scatterplot(x=data['SquareFootage'], y=data['Price'], ax=axs[0])
axs[0].set_title('Square Footage vs. Price')
sns.scatterplot(x=data['Bedrooms'], y=data['Price'], ax=axs[1])
axs[1].set_title('Bedrooms vs. Price')
sns.scatterplot(x=data['Bathrooms'], y=data['Price'], ax=axs[2])
axs[2].set_title('Bathrooms vs. Price')

plt.tight_layout()
plt.show()


# Prediction vs. Actual Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs. Predicted Prices')
plt.show()


# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.show()