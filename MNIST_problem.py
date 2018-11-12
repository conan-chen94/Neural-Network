# Importing the NeuralNetwork class.
from neuralNetwork import NeuralNetwork

# Importing the numpy packages so I have access to arrays, matrices, etc.
import numpy as np

# Importing the pandas package so I can read in excel files.
import pandas as pd

# Importing time to time execution - https://pythonhow.com/measure-execution-time-python-code/
import time

# Starting the MNIST timer.
MNIST_start = time.time()

# Importing excel csv data for the MNIST digit recognizer dataset.
MNIST_train_df = pd.read_csv('MNIST_train.csv')
MNIST_train = MNIST_train_df.values

MNIST_train_ans_set_vector = MNIST_train[:, 0:1]  # first column; Y
MNIST_train_ans_set = np.zeros(shape=(MNIST_train_ans_set_vector.shape[0], 10))
for i in range(MNIST_train_ans_set_vector.shape[0]):
    loc = MNIST_train_ans_set_vector[i, 0]  # the image label
    MNIST_train_ans_set[i, loc] = 1
MNIST_train_inp_set_temp = MNIST_train[:, 1:] / 255  # input matrix; X
MNIST_train_inp_set = np.ceil(MNIST_train_inp_set_temp)

eta = 1

# Creating a neural network object.
MNIST_nn = NeuralNetwork(784, 30, 10)

# Training the neural network.
MNIST_nn.train(MNIST_train_ans_set, MNIST_train_inp_set, 50, eta, .05)

# Validating the neural network
MNIST_test_df = pd.read_csv('MNIST_test.csv')
MNIST_test = MNIST_test_df.values  # no labels in the test set
MNIST_test_inp_set_temp = MNIST_test / 255  # input matrix; X
MNIST_test_inp_set = np.ceil(MNIST_test_inp_set_temp)

MNIST_test_inp_set_transpose = np.transpose(MNIST_test_inp_set)

# Loop through and feedforward every image to generate the prediction.
MNIST_results = np.empty(shape=(MNIST_test_inp_set.shape[0], 2))
for i in range(MNIST_test_inp_set.shape[0]):
    x = MNIST_test_inp_set_transpose[:, i:i+1]  # Use concrete range for the second index to preserve dimensionality
    y_hat = MNIST_nn.feedforward(x)[3]
    y_hat_prediction = np.where(y_hat == np.amax(y_hat))
    MNIST_results[i, 0] = i+1
    MNIST_results[i, 1] = y_hat_prediction[0]

MNIST_results_df = pd.DataFrame(data=MNIST_results, columns=['ImageID', 'Label'])
MNIST_results_df.to_csv('MNIST_submission.csv', index=False)

# Printing the execution time
MNIST_end = time.time()
print('Execution time (s): ', MNIST_end-MNIST_start, '\n')
