import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()

X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)

print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

cm = confusion_matrix(y_test, y_pred)

print('Confusion Matrix:\n', cm)


# Effect of Different K Values

k_values = range(1, 16)
scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)

    scores.append(accuracy_score(y_test, pred))

plt.plot(k_values, scores, marker='o')
plt.xlabel('K value')
plt.ylabel('Test Accuracy')
plt.title('KNN Accuracy for Different K Values')
plt.grid(True)
plt.show()