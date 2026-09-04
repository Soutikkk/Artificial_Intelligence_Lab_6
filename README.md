# 🤖 Artificial Intelligence Lab 6

This repository contains the Python programs for **Artificial Intelligence Lab – 6**, focusing on two important supervised machine learning classification algorithms:

* **K-Nearest Neighbors (KNN)**
* **Support Vector Machine (SVM)**

The programs use the **Iris dataset** provided by Scikit-learn and demonstrate data preprocessing, model training, prediction, and performance evaluation.

---

## 📁 Repository Structure

```text
Artificial_Intelligence_Lab_6/
│
├── KNN Program.py
├── Optional KNN Program.py
├── SVM Program.py
├── SVM — Comparing Linear and RBF Kernels.py
└── README.md
```

---

## 🧠 Programs Included

### 1. `KNN Program.py`

Implements a basic **K-Nearest Neighbors (KNN)** classifier using the Iris dataset.

The program:

* Loads the Iris dataset
* Splits the dataset into training and testing sets
* Standardizes the features using `StandardScaler`
* Trains a KNN classifier with `k = 5`
* Makes predictions on the test dataset
* Calculates classification accuracy
* Displays a classification report
* Generates a confusion matrix

---

### 2. `Optional KNN Program.py`

This program provides an extended implementation of KNN.

In addition to the basic KNN workflow, it evaluates the effect of different values of **K** on test accuracy.

It tests:

```text
K = 1, 2, 3, ..., 15
```

and plots a graph showing:

> **K value vs Test Accuracy**

This helps visualize how the choice of `K` affects the performance of the KNN classifier.

---

### 3. `SVM Program.py`

Implements a **Support Vector Machine (SVM)** classifier using the Iris dataset.

The program:

* Loads the Iris dataset
* Splits the data into training and testing sets
* Standardizes the features
* Uses an SVM with an **RBF (Radial Basis Function) kernel**
* Trains the classifier
* Makes predictions
* Calculates accuracy
* Displays a classification report
* Displays the confusion matrix

The SVM configuration used is:

```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale'
)
```

---

### 4. `SVM — Comparing Linear and RBF Kernels.py`

This program compares two SVM kernels:

* **Linear Kernel**
* **RBF Kernel**

It trains separate SVM models using both kernels and compares their test accuracy.

The comparison is performed using:

```python
kernels = ['linear', 'rbf']
```

The accuracy of each kernel is printed to the console.

---

## 📊 Dataset

All programs use the **Iris dataset** available through Scikit-learn.

The dataset contains measurements of iris flowers based on four features:

* Sepal length
* Sepal width
* Petal length
* Petal width

The target contains three classes:

```text
Setosa
Versicolor
Virginica
```

The dataset is loaded using:

```python
from sklearn.datasets import load_iris

iris = load_iris()
```

---

## 🛠️ Technologies Used

* **Python 3**
* **NumPy**
* **Pandas**
* **Matplotlib**
* **Scikit-learn**

### Install Dependencies

Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## ▶️ How to Run

### Step 1: Clone or extract the repository

Navigate to the project directory:

```bash
cd Artificial_Intelligence_Lab_6
```

### Step 2: Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

### Step 3: Run any program

For example:

```bash
python "KNN Program.py"
```

To run the optional KNN implementation:

```bash
python "Optional KNN Program.py"
```

To run the SVM program:

```bash
python "SVM Program.py"
```

To compare Linear and RBF kernels:

```bash
python "SVM — Comparing Linear and RBF Kernels.py"
```

> **Note:** The quotation marks are useful when running files whose names contain spaces or special characters.

---

## 🔬 Machine Learning Workflow

The programs generally follow this machine learning workflow:

```text
Load Dataset
     ↓
Explore Dataset
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Model Training
     ↓
Prediction
     ↓
Accuracy Evaluation
     ↓
Classification Report
     ↓
Confusion Matrix
```

---

## 📈 Evaluation Metrics

The programs evaluate the trained models using:

### Accuracy

Measures the proportion of correctly classified samples.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Classification Report

Provides:

* Precision
* Recall
* F1-score
* Support

### Confusion Matrix

Shows the number of correct and incorrect predictions for each class.

---

## 🔍 KNN vs SVM

| Feature        | KNN                 | SVM                    |
| -------------- | ------------------- | ---------------------- |
| Type           | Supervised Learning | Supervised Learning    |
| Algorithm      | Instance-based      | Margin-based           |
| Main Parameter | `n_neighbors`       | `C`, `kernel`, `gamma` |
| Scaling        | Important           | Important              |
| Dataset        | Iris                | Iris                   |
| Classification | Yes                 | Yes                    |

The KNN and SVM implementations in this repository provide a practical comparison of two different approaches to classification.

---

## 🎯 Learning Objectives

After completing these programs, you should be able to:

* Understand the basics of **KNN classification**
* Understand the fundamentals of **Support Vector Machines**
* Perform train-test splitting
* Apply feature standardization
* Train Scikit-learn classification models
* Make predictions using trained models
* Evaluate classification performance
* Interpret classification reports
* Understand confusion matrices
* Experiment with different K values in KNN
* Compare Linear and RBF kernels in SVM

---

## 📌 Key Concepts

### K-Nearest Neighbors

KNN classifies a new data point based on the classes of its nearest neighboring data points.

The value of `K` determines how many neighbors are considered.

### Support Vector Machine

SVM attempts to find an optimal decision boundary that separates different classes.

The **RBF kernel** can model non-linear decision boundaries, while the **Linear kernel** is suitable when the classes can be separated approximately linearly.

---

## 👨‍💻 Author

**Artificial Intelligence Lab – 6**

This repository is intended for educational and academic purposes.
