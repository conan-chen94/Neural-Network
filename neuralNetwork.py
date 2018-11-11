# This is the main script for implementing a neural network with an input layer, a hidden layer, and an output layer.

# Importing the numpy packages so I have access to arrays, matrices, etc.
import numpy as np
# Importing the pandas package so I can read in excel files.
import pandas as pd
# Fastest sigmoid function - https://stackoverflow.com/questions/43024745/applying-a-function-along-a-numpy-array
from scipy.special import expit
# Importing time to time execution - https://pythonhow.com/measure-execution-time-python-code/
import time


# Writing out the Neural Network class
class NeuralNetwork:

    # Constructor function
    def __init__(self, num_inputs, num_hidden, num_outputs):
        # Initializing the weights and biases at values within [-0.5,0.5).
        self.weightsIH = np.random.random((num_hidden, num_inputs)) - 0.5
        self.weightsHO = np.random.random((num_outputs, num_hidden)) - 0.5
        self.biasH = np.random.random((num_hidden, 1)) - 0.5
        self.biasO = np.random.random((num_outputs, 1)) - 0.5

        # Initializing the weights and biases at zero.
        # self.weightsIH = np.zeros(shape=(num_hidden, num_inputs))
        # self.weightsHO = np.zeros(shape=(num_outputs, num_hidden))
        # self.biasH = np.zeros(shape=(num_hidden, 1))
        # self.biasO = np.zeros(shape=(num_outputs, 1))

    def train(self, ans_set, inp_set, epochs=5, eta=0.5, threshold=0.001):

        # Transposing ans_set (Y) and inp_set (X) so y and x are naturally column vectors while iterating through.
        ans_set_transpose = np.transpose(ans_set)
        inp_set_transpose = np.transpose(inp_set)

        # Looping through epochs.
        for epoch in range(epochs):

            print('Epoch number: ', epoch)
            total_sum_sq_err = 0

            # Looping through the training cases.
            iterations = range(inp_set.shape[0])
            for i in iterations:

                # Gathering x and y.
                x = inp_set_transpose[:, i:i+1]
                y = ans_set_transpose[:, i:i+1]

                # Feeding forward and calculating sum squared error.
                z_hid, a_hid, z_out, y_hat = self.feedforward(x)
                total_sum_sq_err = total_sum_sq_err + NeuralNetwork.error(y, y_hat)

                # Calculating the output node delta.
                delta = np.multiply((y - y_hat), NeuralNetwork.sigmoid(z_out, True))  # this is (y-y_hat)*o'(z_out)

                # Calculating the new weights and biases.
                temp_weights_ho = self.weightsHO + eta * np.matmul(delta, np.transpose(a_hid))
                temp_weights_ih = self.weightsIH + eta * np.matmul(
                    np.multiply(np.matmul(np.transpose(self.weightsHO), delta), NeuralNetwork.sigmoid(z_hid, True)),
                    np.transpose(x))
                temp_bias_o = self.biasO + eta * delta
                temp_bias_h = self.biasH + eta * np.multiply(np.matmul(np.transpose(self.weightsHO), delta),
                                                             NeuralNetwork.sigmoid(z_hid, True))

                # Updating all the weights.
                self.weightsHO = temp_weights_ho
                self.weightsIH = temp_weights_ih
                self.biasO = temp_bias_o
                self.biasH = temp_bias_h
            mean_sum_sq_err = total_sum_sq_err/inp_set.shape[0]
            print('Mean Sum Squared Error:', mean_sum_sq_err)

            # If the sum squared error falls below our threshold, quit out.
            if mean_sum_sq_err < threshold:
                break

    def validate(self, ans_set, inp_set):

        total_correct = 0
        total_tested = inp_set.shape[0]
        iterations = range(inp_set.shape[0])

        # Transposing ans_set (Y) and inp_set (X) so y and x are naturally column vectors while iterating through.
        ans_set_transpose = np.transpose(ans_set)
        inp_set_transpose = np.transpose(inp_set)

        for i in iterations:
            x = inp_set_transpose[:, i:i + 1]
            y = ans_set_transpose[:, i:i + 1]
            z_hid, a_hid, z_out, y_hat = self.feedforward(x)
            if np.array_equal(y_hat.round(), y):
                total_correct += 1

        return total_correct / total_tested

    def feedforward(self, x):
        z_hid = np.matmul(self.weightsIH, x) + self.biasH
        a_hid = expit(z_hid)
        z_out = np.matmul(self.weightsHO, a_hid) + self.biasO
        y_hat = expit(z_out)
        return z_hid, a_hid, z_out, y_hat

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return np.multiply(expit(x), (1 - expit(x)))
        else:
            return expit(x)

    @staticmethod
    def error(y, y_hat):
        error = y-y_hat
        sq_error = error * error
        sum_sq_error = sq_error.sum()
        return sum_sq_error


# Starting off the timer
start = time.time()

# Importing the excel data for the XOR problem. Pandas dataframe.
df = pd.read_excel('XOR_dataset.xlsx', 'Sheet1')

# Numpy array - https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array-preserving-index
data = df.values
ans_set = data[:, 0:1]  # first column; Y
inp_set = data[:, 1:]  # input matrix; X

# Defining a learning rate
eta = 0.5

# Creating a neural network object.
nn = NeuralNetwork(2, 6, 1)

# Training the neural network.
nn.train(ans_set, inp_set, 15000, eta)

# Validating the neural network.
print('\nSuccess rate: ', nn.validate(ans_set, inp_set))

# Printing the execution time
end = time.time()
print('Execution time (s): ', end-start, '\n')

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

eta = 0.5

# Creating a neural network object.
MNIST_nn = NeuralNetwork(784, 30, 10)

# Training the neural network.
MNIST_nn.train(MNIST_train_ans_set, MNIST_train_inp_set, 1, eta, .05)

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
