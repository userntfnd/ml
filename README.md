# ml
# Expt 9 pca
```
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Compute mean of the data
        self.mean = np.mean(X, axis=0)

        # Center the data
        centered_data = X - self.mean

        # Compute covariance matrix
        cov_matrix = np.cov(centered_data, rowvar=False)

        # Compute eigenvectors and eigenvalues
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvectors based on eigenvalues
        sorted_indices = np.argsort(eigenvalues)[::-1]
        sorted_eigenvectors = eigenvectors[:, sorted_indices]

        # Select top n_components eigenvectors
        self.components = sorted_eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Center the data
        centered_data = X - self.mean

        # Project data onto principal components
        transformed_data = np.dot(centered_data, self.components)

        return transformed_data

# Example usage
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

pca = PCA(n_components=2)
pca.fit(data)
transformed_data = pca.transform(data)

print("Original Data:")
print(data)
print("\nTransformed Data:")
print(transformed_data)



from sklearn.decomposition import PCA
import numpy as np

# Sample data
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Create PCA instance
pca = PCA(n_components=2)

# Fit the model with data
pca.fit(data)

# Transform the data onto the new feature space
transformed_data = pca.transform(data)

print("Original Data:")
print(data)
print("\nTransformed Data:")
print(transformed_data)
```

# expt 4 SVM
```
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

cancer = load_breast_cancer()
X = cancer.data  # 30 Features
y = cancer.target  # Target(2 Classes)

cancer_df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)

print(cancer_df.head(5))
print(cancer_df.shape)

print("Type of dataset:") 1
print(type(cancer))

print("\nTarget values:")
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.svm import SVC

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
print("\nPredicted values:")
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

precision = precision_score(y_test, y_pred)
print("Precision:", precision)

recall = recall_score(y_test, y_pred)
print("Recall:", recall)

f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)
```

# expt 8  perceptron

```
import numpy as np

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.zeros(input_size + 1)  # +1 for bias

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]  # w.x + b
        return self.activation(summation)

    def train(self, training_inputs, labels, learning_rate=0.01, epochs=100):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += learning_rate * (label - prediction) * inputs
                self.weights[0] += learning_rate * (label - prediction)

# Example usage
training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size=2)
perceptron.train(training_inputs, labels)

test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
for inputs in test_inputs:
    print(inputs, perceptron.predict(inputs))

```

# Expt 7 Mp neuron
```
import numpy as np

x1=np.array([0,0,1,1])
x2=np.array([0,1,0,1])
t=[0,0,0,1]

w1=float(input("Enter W1 :"))
w2=float(input("Enter W2 :"))
T=float(input("Enter Threshold "))

yinit = (w1*x1) + (w2*x2)
print("Yinitial:",yinit)

y=np.array([0,0,0,0])
for i in range(4):
  if yinit[i]>=T:
    y[i]=1
  else:
    y[i]=0

print("")
print("t",t)
print("y",y)
print("")
if np.array_equal(y,t):
  print("Correct Weight and Threshold Values")
else:
  print("Incorrect Weights please Re-run Code")

```
