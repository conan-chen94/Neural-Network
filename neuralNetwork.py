# This is the main script for implementing a neural network with an input layer, a hidden layer, and an output layer.

# Importing the numpy packages so I have access to arrays, matrices, etc.
import numpy as np
# Fastest sigmoid function - https://stackoverflow.com/questions/43024745/applying-a-function-along-a-numpy-array
from scipy.special import expit


# Writing out the Neural Network class
class NeuralNetwork:

    # Constructor function.
    def __init__(self, num_inputs, num_hidden, num_outputs):
        # Initializing the weights and biases at values within [-0.5,0.5).
        self.weightsIH = np.random.random((num_hidden, num_inputs)) - 0.5
        self.weightsHO = np.random.random((num_outputs, num_hidden)) - 0.5
        self.biasH = np.random.random((num_hidden, 1)) - 0.5
        self.biasO = np.random.random((num_outputs, 1)) - 0.5

    # Training function.
    def train(self, ans_set, inp_set, epochs=5, eta=0.5, threshold=0.001):

        # Transposing ans_set (Y) and inp_set (X) so y and x are naturally column vectors while iterating through.
        ans_set_transpose = np.transpose(ans_set)
        inp_set_transpose = np.transpose(inp_set)

        # Calculating and printing the statistics for the beginning state.
        y_hat_set = self.feedforward(inp_set_transpose)[3]
        total_sum_sq_err_set = NeuralNetwork.error(ans_set_transpose, y_hat_set)
        mean_sum_sq_err = total_sum_sq_err_set.sum()/inp_set.shape[0]

        print('Epoch number: ', str(0).ljust(15), end="")
        print('Mean Sum Squared Error: ', format(mean_sum_sq_err, '.10f').ljust(25), end="")
        print('Success Rate: ', format(self.validate(ans_set, inp_set), '.10f'))

        # Looping through epochs.
        for epoch in range(1, epochs+1):

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

            # Printing the statistics for each epoch.
            print('Epoch number: ', str(epoch).ljust(15), end="")
            print('Mean Sum Squared Error: ', format(mean_sum_sq_err, '.10f').ljust(25), end="")
            print('Success Rate: ', format(self.validate(ans_set, inp_set), '.10f'))

            # If the sum squared error falls below our threshold, quit out.
            if mean_sum_sq_err < threshold:
                break

    # Validation function.
    def validate(self, ans_set, inp_set):

        total_correct = 0
        total_tested = inp_set.shape[0]
        iterations = range(inp_set.shape[0])

        # Transposing ans_set (Y) and inp_set (X) so y and x are naturally column vectors while iterating through.
        ans_set_transpose = np.transpose(ans_set)
        inp_set_transpose = np.transpose(inp_set)

        # Generating the predictions.
        y_hat_set = self.feedforward(inp_set_transpose)[3]
        y_hat_set_round = y_hat_set.round()

        # Counting the correct predictions.
        for i in iterations:
            y = ans_set_transpose[:, i:i+1]
            y_hat = y_hat_set_round[:, i:i+1]
            if np.array_equal(y_hat, y):
                total_correct += 1

        return total_correct / total_tested

    # Feed forward function.
    def feedforward(self, x):
        z_hid = np.matmul(self.weightsIH, x) + self.biasH
        a_hid = expit(z_hid)
        z_out = np.matmul(self.weightsHO, a_hid) + self.biasO
        y_hat = expit(z_out)
        return z_hid, a_hid, z_out, y_hat

    # Static method for the sigmoid function.
    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return np.multiply(expit(x), (1 - expit(x)))
        else:
            return expit(x)

    # Static method for calculating the sum squared error of a set of outputs.
    @staticmethod
    def error(y, y_hat):
        error = y-y_hat
        sq_error = error * error
        sum_sq_error = sq_error.sum()
        return sum_sq_error