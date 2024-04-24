Expt 1 keras
```
#importing libraries
import tensorflow as tf
import matplotlib.pyplot as plt

#obtaining dataset
mnist = tf.keras.datasets.mnist

#loading the dataset
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#obtaining the size
# plt.imshow(x_train[8])
x_train.shape

#data obtained from dataset
x_train[8]

#preproceesing
x_train , x_test = x_train/255.0 , x_test/255.0

# 1) Defining the Network (Model)
model = tf.keras.models.Sequential([

  tf.keras.layers.Flatten(input_shape = (28,28)),
  tf.keras.layers.Dense(70, activation = 'relu'),
  tf.keras.layers.Dense(10 , activation= 'softmax')
])

#2) Compliling the Network (Model)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
model.compile(
    optimizer = 'SGD',
    loss = loss_fn,
    metrics = ['accuracy'])

#3) Fitting the Network (Model)
model.fit(x_train,y_train, epochs = 13)

#4) Evaluating the Network (Model)
model.evaluate(x_test,y_test)
```

Expt 2 linear regression
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset =pd.read_csv('C:/Users/Zaurez/OneDrive/Documents/Desktop/to upload/Salary_dataset ML - Salary_dataset.csv')

X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values

Z=X*Y
X2 = X*X
A = (len(X)*sum(Z)) - (sum(X)*sum(Y))  / (len(X)*sum(X2)) - (sum(X)*sum(X))
B =  (sum(Y) - (A*sum(X)) / len(X))

y_pred = []
for i in range (len(X)):
    y_pred.append((A*X) + B)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X.reshape(-1,1),Y)

plt.plot(X.reshape(-1,1),regressor.predict(X.reshape(-1,1)) , color = 'blue' , label = 'best fit line')
plt.scatter(X,Y,color ='red',label ="Predicted data")
plt.title
plt.xlabel
plt.ylabel
plt.show
```

Expt 3 logistic regression
```
import numpy as np

b0=0
b1=0
b2=0

n = int(input('Enter the number of values:'))
x1 = list(map(float ,input('Enter the number of values:').split(',')))
x1= np.array(x1)
x2 = list(map(float ,input('Enter the number of values:').split(',')))
x2= np.array(x2)
y = list(map(float ,input('Enter the number of values:').split(',')))
y= np.array(y)
S=0.5

p = []
pc =[]

for i in range(min(n, len(x1), len(x2), len(y))):
  p_e = (1/(1+round((np.exp(-(b0+b1*x1[i]+b2*x2[i]))),2)))
  b0 += S*(y[i]-p_e) * p_e * (1-p_e) * 1
  b1 += S*(y[i]-p_e) * p_e * (1-p_e) * x1[i]
  b2 += S*(y[i]-p_e) * p_e * (1-p_e) * x2[i]
  p.append(p_e)
  if p_e >= S:
    pc.append(1)
  else:
    pc.append(0)


print("the Predictions and Predicted classes are:")
for i in range (len(p)):
  print(f"x1 : {x1[i]} , x2: {x2[i]} , y: {y[i]} , Predictions : {p[i]} , predicted class : {pc[i]} ")
```
expt 4 SVM
```
from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load breast cancer dataset
data = datasets.load_breast_cancer()
X = data.data
y = data.target

# Print dataset information
print("Number of classes:", len(set(y)))
print("Number of samples per class:")
print(pd.Series(y).value_counts())
print("Total number of samples:", len(y))
print("Dimensionality:", X.shape[1])
print("Features:", data.feature_names)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=60)

# Initialize and train SVM classifier
object_SVM = SVC(kernel='linear')
object_SVM.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = object_SVM.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
```
expt 5 hebbian learning
```
def heb_learning(samples):
  print(f'{"Input" : ^8}{"Target" : ^16}{"Weight changes" : ^15}{"Weight" : ^28}')
  w1 , w2 , b = 0 , 0 , 0
  print(' ' * 48 , f'({w1:2} , {w2:2} , {b:2})')
  for x1,x2,y in samples:
    w1 = w1 + x1 * y
    w2 = w2 + x2 * y
    b=b+y
    print(f'({x1:2} , {x2:2})\t {y:2}\t ({x1*y:2},{x2*y:2},{y:2})\t\t ({w1:2} , {w2:2} , {b:2})')

OR_samples = {
    'bipolar_input_bipolar_output' : [
        [1,1,1],
        [1,-1,1],
        [-1,1,1],
        [-1,-1,-1]
    ]
}

print('-'*20,'Hebbian Learning','-'*20)

print('OR with bipolar input and bipoloar output')
heb_learning(OR_samples['bipolar_input_bipolar_output'])

def heb_learnings(samples):
  print(f'{"Input" : ^6}{"Target" : ^12}{"Changes" : ^12}{"Initial" : ^12}')
  w1=0
  w2=0
  b=0
  print(' '*32, f'{w1:2}{w2:2}{b:2}')
  for x1,x2,y in samples:
    w1 = w1 + x1*y
    w2 = w2 + x2*y
    b = b+y
    print(f'{x1:2}{x2:2}\t {y:2}\t {x1*y:2}{x2*y:2}{y:2}\t\t {w1:2}{w2:2}{b:2}')

AND_samples = {

    'bipolar_input_bipolar_output' : [

          [1,1,1],
          [1,-1,-1],
          [-1,1,-1],
          [-1,-1,-1]
    ]
}

print("AND gate hebbian Learning\n")
heb_learnings(AND_samples['bipolar_input_bipolar_output'])
```
expt 7 mp neuron
```
import numpy as np

while True :
  x1 = np.array([0, 0, 1, 1])
  x2 = np.array([0,  1, 0, 1])
  t = np.array([0, 0, 1, 0])
  w1 = float(input("Enter W1 Weight Value: "))
  w2 = float(input("Enter W2 Weight Value: "))
  T = float(input("Enter Threshold Value: "))


  yin = w1 * x1 + w2 * x2

  print("Yin:")
  print(yin)

  y = np.zeros_like(t) #to make 0 for -ve values

  for i in range(len(yin)):
     if yin[i] >= T:
         y[i] = 1

  print("")

  print("Target O/P", t)
  print("Calculated O/P", y)
  print("")

  if np.array_equal(y, t):
    print("Correct Weight And Threshold Values")
    break
  else:
    print("Incorrect Weights, Re-running Code")
    print("\n")
```

expt 8 perceptron model 
```
import numpy as np

# Input array and desired output
input_array = np.array([1, 1, 0, 1])
desired_output = 1

# Define parameters
input_weights = np.array([0.3, -0.2, 0.2, 0.1])
bias = 0.2
learning_rate = 0.8

# Initialize iteration counter
iteration = 0
max_iterations = int(input("Enter the maximum number of iterations for the algorithm: "))

while iteration < max_iterations:
    # Forward pass
    net_input = np.dot(input_array, input_weights) + bias
    output = 1 / (1 + np.exp(-net_input))

    # Error calculation
    error = desired_output - output

    # Update weights and bias
    input_weights += learning_rate * error * input_array
    bias += learning_rate * error

    # Increment iteration counter
    iteration += 1

print("Weights:", input_weights)
print("Bias:", bias)
print("Number of iterations executed:", iteration)
print("Final Error:", error)

def perceptron_learning(samples):
    print(f'{"Input":^8}{"Target":^16}{"Weight changes":^15}{"Weights":^28}')
    w1, w2, b = 0, 0, 0
    print(' ' * 48, f'({w1:2}, {w2:2}, {b:2})')
    for x1, x2, y in samples:
        # Calculate the predicted output
        output = 1 if w1 * x1 + w2 * x2 + b >= 0 else 0

        # Update weights and bias using perceptron learning rule
        w1_change = x1 * (y - output)
        w2_change = x2 * (y - output)
        b_change = y - output

        w1 += w1_change
        w2 += w2_change
        b += b_change

        print(f'({x1:2}, {x2:2})\t {y:2}\t ({w1_change:2}, {w2_change:2}, {b_change:2})\t\t ({w1:2}, {w2:2}, {b:2})')

AND_samples = {
    'binary_input_binary_output': [
        [1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
}

print('-' * 20, 'Perceptron Learning', '-' * 20)
print('AND with binary input and binary output')
perceptron_learning(AND_samples['binary_input_binary_output'])
```
expt 9  PCA
```
import numpy as np
import matplotlib.pyplot as plt

# Define dataset
x = np.array([4, 8, 13, 7])
y = np.array([11, 4, 5, 14])
dataset = np.array([x, y])
print("Define Dataset")
print(dataset)
print()

# Finding mean
xMean = np.mean(x)
yMean = np.mean(y)

# Adjusting the mean obtained
MeanAdjusted = np.array([x - xMean, y - yMean])
print("Mean adjusted:")
print(MeanAdjusted)
print("\n")

# Finding the Covariance
covariance_matrix = np.cov(dataset)
print("Covariance Matrix")
print(covariance_matrix)
print("\n")

# Compute the Eigen Values and Eigen Vectors
eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
print("Eigen Values")
print(eigen_values)
print()

# Sort result in descending order
sorted_indices = np.argsort(eigen_values)[::-1]
sorted_eigen_values = eigen_values[sorted_indices]
sorted_eigen_vectors = eigen_vectors[:, sorted_indices]
print("Sorted Eigen Values")
print(sorted_eigen_values)
print()
print("Sorted Eigen Vectors")
print(sorted_eigen_vectors)
print()

# Perform PCA
PCA = np.dot(sorted_eigen_vectors.T, MeanAdjusted)
print("Principal Component Analysis:")
print(PCA)

# Scatter plot after PCA
plt.subplot(1, 2, 2)
plt.scatter(PCA[0], PCA[1], color='red')
plt.title('After PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.tight_layout()
plt.show()
```
