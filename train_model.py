import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("house_data.csv")

X = data[['Area', 'Bedrooms']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predicted_price = model.predict([[1400, 3]])
print(predicted_price[0])

y_pred = model.predict(X_test)

print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))

X_area = data[['Area']]
y_actual = data['Price']

model_area = LinearRegression()
model_area.fit(X_area, y_actual)

X_sorted = X_area.sort_values(by='Area')
y_pred_line = model_area.predict(X_sorted)

plt.figure(figsize=(8, 5))
plt.scatter(X_area, y_actual)
plt.plot(X_sorted, y_pred_line)
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.grid(True)
plt.show()
