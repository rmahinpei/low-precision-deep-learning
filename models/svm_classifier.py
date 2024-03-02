# Implementation of an SVM classifier supporting multi-precision training
# Implementation currently supports training in either double, single, or half precision
# Implementation is based off of scikit-learn's very own example: 
# https://scikit-learn.org/stable/auto_examples/svm/plot_iris_svc.html

import time
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Define number of training runs to compute the average training time over
NUM_TRAINING_RUNS = 5
 
def train(X_train, y_train, precision):
    if precision == 'double':
        dtype = np.double
    elif precision == 'single':
        dtype = np.single
    else: # half
        dtype = np.half
    
    X_train = np.array(X_train, dtype=dtype)
    y_train = np.array(y_train, dtype=dtype)
    model = SVC(kernel='linear')

    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    
    return model, training_time 

# Load dataset and split into train and test sets
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=12)

# Test run to make sure that everything is working properly before starting actual measurements
_ = train(X_train, y_train, precision='single')

# Train and evaluate with double precision
time_double = 0.0
for _ in range(NUM_TRAINING_RUNS):
    model_double, training_time = train(X_train, y_train, precision='double')
    time_double += training_time
accuracy_double = model_double.score(X_test, y_test)

# Train and evaluate with single precision
time_single = 0.0
for _ in range(NUM_TRAINING_RUNS):
    model_single, training_time = train(X_train, y_train, precision='single')
    time_single += training_time
accuracy_single = model_single.score(X_test, y_test)

# Train and evaluate with half precision
time_half = 0.0
for _ in range(NUM_TRAINING_RUNS):
    model_half, training_time = train(X_train, y_train, precision='half')
    time_half += training_time
accuracy_half = model_half.score(X_test, y_test)

print("---RESULTS---")
print("Average training time in double precision:", time_double / NUM_TRAINING_RUNS, "seconds")
print("Average training time in single precision:", time_single/ NUM_TRAINING_RUNS, "seconds")
print("Average training time in half   precision:", time_half/ NUM_TRAINING_RUNS, "seconds")
print("-------------")
print("Accuracy with double precision:", accuracy_double)
print("Accuracy with single precision:", accuracy_single)
print("Accuracy with half   precision:", accuracy_single)

