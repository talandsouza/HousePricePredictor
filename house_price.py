import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate some example data
np.random.seed(0)
num_samples = 100
bedrooms = np.random.randint(1, 6, num_samples)  # Simulating the number of bedrooms
house_prices = 50000 + 25000 * bedrooms + np.random.normal(0, 10000, num_samples)  # Simulating house prices

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(bedrooms.reshape(-1, 1), house_prices, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print the coefficients and intercept
print("Coefficients:", model.coef_)  # Slope of the line
print("Intercept:", model.intercept_)  # Y-intercept

# Plot the actual vs. predicted prices
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')  # Scatter plot of actual prices
plt.plot(X_test, predictions, color='red', label='Predicted Prices')  # Line plot of predicted prices
plt.xlabel('Number of Bedrooms')
plt.ylabel('House Price')
plt.title('House Price Prediction')
plt.legend()
plt.show()  # Display the plot


