```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
```

```python
# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
```

```python
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

```python
# Initialize and fit the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.labels_
```

```python
# Map cluster labels to true labels to calculate accuracy
# Since KMeans assigns arbitrary labels, we need to match them to actual labels
from scipy.stats import mode

labels = np.zeros_like(clusters)
for i in range(3):
    mask = (clusters == i)
    labels[mask] = mode(y[mask])[0]

accuracy = accuracy_score(y, labels)
print(f"K-Means Clustering Accuracy: {accuracy * 100:.2f}%")
```

```output
K-Means Clustering Accuracy: 66.67%

```

```python

```

