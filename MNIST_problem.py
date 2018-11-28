# Importing the NeuralNetwork class.
from neuralNetwork import NeuralNetwork

# Importing the numpy packages so I have access to arrays, matrices, etc.
import numpy as np

# Importing the pandas package so I can read in excel files.
import pandas as pd

# Importing time to time execution - https://pythonhow.com/measure-execution-time-python-code/
import time

# Importing pickle to save any variables as needed.
import pickle

# Starting the MNIST timer.
MNIST_start = time.time()

# Importing excel csv data for the MNIST digit recognizer dataset.
MNIST_train_df = pd.read_csv('MNIST_train.csv')
MNIST_train = MNIST_train_df.values


# This function takes in a set of labels and transforms those labels into one-hot vectors. For example, transforms a
# label of 2 into [0, 0, 1, 0, 0, 0, 0, 0, 0, 0].
def one_hot_encode(labels):
    num_labels = labels.shape[0]
    vector_set = np.zeros(shape=(num_labels, 10))
    for i in range(num_labels):
        loc = labels[i, 0]  # the image label
        vector_set[i, loc] = 1
    return vector_set


# This function takes in a set of output vectors and transforms those vectors in labels. The output vectors are one-hot-
# like, but not exactly one-hot.
def one_hot_decode(vector_set):
    num_labels = vector_set.shape[0]
    labels = np.empty(shape=(num_labels, 1))
    for i in range(num_labels):
        vector = vector_set[i, :]
        labels[i, 0] = np.where(vector == np.amax(vector))[0]
    return labels


# Encoding the labels into one-hot vectors.
MNIST_train_ans_set = one_hot_encode(labels=MNIST_train[:, 0:1])

# Pre-processing the images so that, instead of having pixel values in [0, 255], they are normalized to [0, 1].
MNIST_train_inp_set_temp = MNIST_train[:, 1:] / 255  # input matrix; X
MNIST_train_inp_set = np.ceil(MNIST_train_inp_set_temp)

# Creating a neural network object.
MNIST_nn = NeuralNetwork(num_inputs=784, num_hidden=50, num_outputs=10)

# Defining a learning rate.
eta = 1

# Training the neural network.
MNIST_nn.train(ans_set=MNIST_train_ans_set, inp_set=MNIST_train_inp_set, epochs=10, eta=eta, threshold=.02)

# Using the trained neural network to generate predictions on the test set.
# First, reading in the test set images, and doing the same pre-processing as on the training set.
MNIST_test_df = pd.read_csv('MNIST_test.csv')
MNIST_test = MNIST_test_df.values  # no labels in the test set
MNIST_test_inp_set_temp = MNIST_test / 255  # input matrix; X
MNIST_test_inp_set = np.ceil(MNIST_test_inp_set_temp)

# Transposing the input set so X is a collection of column vectors, each column representing an image.
MNIST_test_inp_set_transpose = np.transpose(MNIST_test_inp_set)

# Loop through and feed forward every image to generate the prediction.

# Use the decoding functions you wrote.
num_results = MNIST_test_inp_set.shape[0]
MNIST_results = np.empty(shape=(num_results, 2))

y_hats = MNIST_nn.feedforward(MNIST_test_inp_set_transpose)[3]
y_hats_transpose = np.transpose(y_hats)
y_hat_predictions = one_hot_decode(y_hats_transpose)
MNIST_results[:, 0] = np.arange(start=1, step=1, stop=num_results+1)
MNIST_results[:, 1] = y_hat_predictions.flatten()


# Writing out results to a file named "MNIST_submission.csv". This is submitted to Kaggle.
MNIST_results_df = pd.DataFrame(data=MNIST_results, columns=['ImageID', 'Label'])
MNIST_results_df.to_csv('MNIST_submission.csv', index=False)

# Printing the execution time
MNIST_end = time.time()
print('\n', 'Execution time (s): ', format(MNIST_end-MNIST_start, '.10f'), '\n')