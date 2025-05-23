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
print(X_train.shape , X_test.shape , y_train.shape , y_test.shape)
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
print(model.intercept_)
```

```output
26767.86524944281

```

```python
print(model.coef_)
```

```output
[9459.35953483]

```

```python
#predict model
y_pred = model.predict(X_test)
```

```python
print(y_pred)
```

```output
[ 69334.98315618  64605.30338877  64605.30338877  83524.02245843
  45686.5843191   61767.49552832 117577.71678382  85415.89436539
 126091.14036516  50416.26408652  82578.08650494  57037.8157609 ]

```

```python
#model accuracy
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error, mean_squared_error
```

```python

print("MSE:",mean_absolute_error(y_test,y_pred))
```

```output
MSE: 5211.981371554558

```

```python
print("MAPE:",mean_absolute_percentage_error(y_test,y_pred))
```

```output
MAPE: 0.0761963556601618

```

```python
print("MSE:",mean_squared_error(y_test,y_pred))
```

```output
MSE: 35344480.17477033

```

```python
y_pred_line = model.predict(X)
y_pred_test = model.predict(X_test)
```

```python

```

```python
# Scatter plot of actual vs predicted
import matplotlib.pyplot as plt
import seaborn as sns

# Plot actual data
plt.figure(figsize=(8, 6))
plt.scatter(X, y, marker = '*', label='Actual Data')                   # all real data
plt.plot(X, y_pred_line, marker ='o', label='Regression Line')         # regression curve
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

```python

```

