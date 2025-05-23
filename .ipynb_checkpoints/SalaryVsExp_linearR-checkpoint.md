```python
import pandas as pd
```

```python
salary = pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Salary%20Data.csv')
```

```python
#definging target(y) and feature(x)
y = salary['Salary']
X = salary[['Experience Years']]
```

```python
#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,train_size = 0.7,random_state =42)
```

```python
#Check shape of train and test sample
X_train.shape , X_test.shape , y_train.shape , y_test.shape
```

```python
#select model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

```python
model.fit(X_train,y_train)
```

```python
model.intercept_
```

```python
model.coef_
```

```python
#predict model
y_pred = model.predict(X_test)
```

```python
y_pred
```

```python
#model accuracy
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error, mean_squared_error
```

```python
mean_absolute_error(y_test,y_pred)
```

```python
mean_absolute_percentage_error(y_test,y_pred)
```

```python
mean_squared_error(y_test,y_pred)
```

```python
y_pred_line = model.predict(X)
y_pred_test = model.predict(X_test)
```

```python
# Scatter plot of actual vs predicted
import matplotlib.pyplot as plt
import seaborn as sns

# Plot actual data
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')                   # all real data
plt.plot(X, y_pred_line, color='red', label='Regression Line')         # regression curve
plt.scatter(X_test, y_pred_test, color='green', label='Predicted Points')  # predicted points on test data

plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression: Actual vs Predicted with Regression Line")
plt.legend()
plt.grid(True)
plt.show()
```

```python

```

