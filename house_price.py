
from sklearn.linear_model import LinearRegression
import numpy as np


X = np.array([[500], [1000], [1500], [2000], [2500]])   
y = np.array([150000, 200000, 250000, 300000, 350000])  


model = LinearRegression()


model.fit(X, y)


new_house = np.array([[15400]])   
predicted_price = model.predict(new_house)

print(f"Predicted price sqft house: ${predicted_price[0]:,.2f}")
