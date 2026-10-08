# Decision Tree Classification Program
# Artificial Intelligence Lab

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Decision Tree model
model = DecisionTreeClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Decision Tree Classification")
print("--------------------------------")
print("Accuracy:", accuracy)

# Display actual and predicted values
print("\nActual Values:   ", y_test)
print("Predicted Values:", y_pred)

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Predict a new flower
new_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(new_flower)

print("\nNew Flower Prediction:")
print("Predicted Class:", iris.target_names[prediction[0]])
