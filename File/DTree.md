```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
```

```python
#readin data
data = pd.read_csv("play_golf_dataset.csv")
```

```python
df = pd.DataFrame(data)
```

```python
#Step2 : Encode Categorical Variables to numbers
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)
```

```python
#Step3 : Separate features and target
X = df_encoded.drop("Play Golf",axis = 1)
y = df_encoded["Play Golf"]
```

```python
from sklearn.model_selection import train_test_split
# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

```

```python
# Step 6: Predict on the test set
y_pred = clf.predict(X_test)
```

```python
from sklearn.metrics import accuracy_score, precision_score
# Step 7: Evaluate Accuracy and Precision
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)  # Default is for class "1" (i.e., "Yes")

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
```

```output
Accuracy: 0.57
Precision: 0.68

```

```python

```

