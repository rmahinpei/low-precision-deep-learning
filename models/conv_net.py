# Implementation of a convolutional nerural network supporting multi-precision training
# Implementation currently supports training in either double, single, or half precision
# Implementation is based off of TensorFlow's very own example:
# https://www.tensorflow.org/tutorials/images/cnn

import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Define number of training runs to compute the average training time over
NUM_TRAINING_RUNS = 5
# Values are specific to the CIFAR10 dataset
INPUT_DIM = 32
INPUT_CHANNELS = 3
NUM_CLASSES = 10

def build_and_train(X_train, y_train, precision='single'):
    if precision == 'double':
        dtype = tf.float64
    elif precision == 'single':
        dtype = tf.float32
    else: # half
        dtype = tf.float16
    
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(INPUT_DIM, INPUT_DIM, INPUT_CHANNELS), dtype=dtype),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', dtype=dtype),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(dtype=dtype),
        layers.Dense(64, activation='relu', dtype=dtype),
        layers.Dense(NUM_CLASSES, activation='softmax', dtype=dtype)
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    start_time = time.time()
    model.fit(X_train, y_train, epochs=5)
    end_time = time.time()
    training_time = end_time - start_time
    
    return model, training_time


# Load dataset and split into train and test sets
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Scale values before feeding into neural net
X_train = X_train / 255.0
X_test = X_test / 255.0

# Test run to make sure that everything is working properly before starting actual measurements
_ = build_and_train(X_train, y_train, precision='single')

# Train with double precision
time_double = 0.0
for _ in range(NUM_TRAINING_RUNS):
    model_double, training_time = build_and_train(X_train, y_train, 'double')
    time_double += training_time
accuracy_double = model_double.evaluate(X_test, y_test, verbose=2)[1]

# Train with single precision
time_single = 0.0
for _ in range(NUM_TRAINING_RUNS):
    model_single, training_time = build_and_train(X_train, y_train, 'single')
    time_single += training_time
accuracy_single = model_single.evaluate(X_test, y_test, verbose=2)[1]

# Train with half precision
time_half = 0.0
for _ in range(NUM_TRAINING_RUNS):
    model_half, training_time = build_and_train(X_train, y_train, 'half')
    time_half += training_time
accuracy_half = model_half.evaluate(X_test, y_test, verbose=2)[1]

print("---RESULTS---")
print("Average training time in double precision:", time_double / NUM_TRAINING_RUNS, "seconds")
print("Average training time in single precision:", time_single/ NUM_TRAINING_RUNS, "seconds")
print("Average training time in half   precision:", time_half/ NUM_TRAINING_RUNS, "seconds")
print("-------------")
print("Accuracy with double precision:", accuracy_double)
print("Accuracy with single precision:", accuracy_single)
print("Accuracy with half   precision:", accuracy_single)