# Implementation of a classic SVM classifier supporting **multi-precision** training
    # Implementation currently supports training in either double, single, or half precision
    # This implies that both the computations and parameter storage are done in the specified precision
# Implementation of a classic SVM network classifier supporting **mixed-precision** training
    # Implementation currently supports half precision computations with single precision parameter storage

import time
import tensorflow as tf
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, utils
from sklearn.model_selection import train_test_split

# Define number of training runs to compute the average training time over
NUM_TRAINING_RUNS = 5
 
def build_and_train(X_train, y_train, precision):
    if precision == 'double':
        dtype = tf.float64
    elif precision == 'single':
        dtype = tf.float32
    else: # half
        dtype = tf.float16

    model = models.Sequential([
        layers.Dense(3, activation='linear', dtype=dtype)  
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    start_time = time.time()
    model.fit(X_train, y_train, epochs=100)
    end_time = time.time()
    training_time = end_time - start_time
    
    return model, training_time

def build_and_train_mixed(X_train, y_train):
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model = models.Sequential([
        layers.Dense(3, activation='linear')  
    ])
    model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

    start_time = time.time()
    model.fit(X_train, y_train, epochs=5)
    end_time = time.time()
    training_time = end_time - start_time

    tf.keras.mixed_precision.set_global_policy('float32')
    return model, training_time

# Load dataset and split into train and test sets
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert labels to one-hot encoding
y_train = utils.to_categorical(y_train)
y_test = utils.to_categorical(y_test)

# Test run to make sure that everything is working properly before starting actual measurements
_ = build_and_train(X_train, y_train, precision='single')
_ = build_and_train_mixed(X_train, y_train)

# Train and evaluate with double precision
time_double = 0.0
for _ in range(NUM_TRAINING_RUNS):
    model_double, training_time = build_and_train(X_train, y_train, precision='double')
    time_double += training_time
accuracy_double = model_double.evaluate(X_test, y_test, verbose=2)[1]

# Train and evaluate with single precision
time_single = 0.0
for _ in range(NUM_TRAINING_RUNS):
    model_single, training_time = build_and_train(X_train, y_train, precision='single')
    time_single += training_time
accuracy_single = model_single.evaluate(X_test, y_test, verbose=2)[1]

# Train and evaluate with half precision
time_half = 0.0
for _ in range(NUM_TRAINING_RUNS):
    model_half, training_time = build_and_train(X_train, y_train, precision='half')
    time_half += training_time
accuracy_half = model_half.evaluate(X_test, y_test, verbose=2)[1]

# Train with mixed half precision
time_mixed = 0.0
for _ in range(NUM_TRAINING_RUNS):
    model_mixed, training_time = build_and_train_mixed(X_train, y_train)
    time_mixed += training_time
accuracy_mixed = model_mixed.evaluate(X_test, y_test, verbose=2)[1]

print("---RESULTS---")
print("Average training time in double precision:", time_double / NUM_TRAINING_RUNS, "seconds")
print("Average training time in single precision:", time_single/ NUM_TRAINING_RUNS, "seconds")
print("Average training time in half precision:", time_half/ NUM_TRAINING_RUNS, "seconds")
print("Average training time in mixed half precision:", time_mixed/ NUM_TRAINING_RUNS, "seconds")
print("-------------")
print("Accuracy with double precision:", accuracy_double)
print("Accuracy with single precision:", accuracy_single)
print("Accuracy with half precision:", accuracy_single)
print("Accuracy with mixed half precision:", accuracy_mixed)

