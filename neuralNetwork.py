# This is the main script for implementing a neural network with an input layer, a hidden layer, and an output layer.

# Importing the numpy packages so I have access to arrays, matrices, etc.
import numpy as np
import pandas as pd
import math
from scipy.special import \
    expit  # Fastest per https://stackoverflow.com/questions/43024745/applying-a-function-along-a-numpy-array
#from keras.models import Sequential
#from keras.layers import Dense, Activation


# Writing out the Neural Network class
class NeuralNetwork:
    # Constructor function
    def __init__(self, num_inputs, num_hidden, num_outputs):
        # self.numInputs = num_inputs
        # self.numHidden = num_hidden
        # self.numOutputs = num_outputs
        self.weightsIH = np.random.random((num_hidden, num_inputs)) - 0.5
        self.weightsHO = np.random.random((num_outputs, num_hidden)) - 0.5
        self.biasH = np.random.random((num_hidden, 1)) - 0.5
        self.biasO = np.random.random((num_outputs, 1)) - 0.5

        #self.weightsIH = np.zeros(shape=(num_hidden, num_inputs))
        #self.weightsHO = np.zeros(shape=(num_outputs, num_hidden))
        #self.biasH = np.zeros(shape=(num_hidden, 1))
        #self.biasO = np.zeros(shape=(num_outputs, 1))

    def train(self, ans_set, inp_set, epochs=5, eta=0.5):

        for epoch in range(epochs):

            print('Epoch number: ', epoch)
            sumSqErr = 0
            iterations = range(inp_set.shape[0])
            for i in iterations:
                x = inp_set[i, :]
                x = x.reshape(x.shape[0], 1)  # reshape as column vector
                y = ans_set[i, :]
                y = y.reshape(y.shape[0], 1)  # reshape as column vector
                z_hid, a_hid, z_out, y_hat = self.feedforward(x)
                sumSqErr = sumSqErr + (y - y_hat) ** 2

                delta = np.multiply((y - y_hat), NeuralNetwork.sigmoid(z_out, True))  # this is (y-y_hat)*o'(z_out)
                temp_weights_ho = self.weightsHO + eta * np.matmul(delta, np.transpose(a_hid))
                temp_weights_ih = self.weightsIH + eta * np.matmul(
                    np.multiply(np.matmul(np.transpose(self.weightsHO), delta), NeuralNetwork.sigmoid(z_hid, True)),
                    np.transpose(x))
                temp_bias_o = self.biasO + eta * delta
                temp_bias_h = self.biasH + eta * np.multiply(np.matmul(np.transpose(self.weightsHO), delta),
                                                             NeuralNetwork.sigmoid(z_hid, True))

                # Fourth, updating everything.
                self.weightsHO = temp_weights_ho
                self.weightsIH = temp_weights_ih
                self.biasO = temp_bias_o
                self.biasH = temp_bias_h

            print('Sum Squared Error:', np.asscalar(sumSqErr))

            if sumSqErr < 0.001:
                break

    def validate(self, dataSet):
        ansSet = dataSet[:, 0]  # first column
        valSet = dataSet[:, 1:]

        totalCorrect = 0
        totalTested = valSet.shape[0]
        iterations = range(valSet.shape[0])

        for i in iterations:
            x = valSet[i, :]
            x = x.reshape(x.shape[0], 1)  # Reformat as column vector
            y = ansSet[i]
            z_hid, a_hid, z_out, y_hat = self.feedforward(x)
            if round(np.asscalar(y_hat)) == y:
                totalCorrect += 1

        return totalCorrect / totalTested

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


# Importing the excel data for the XOR problem. Pandas dataframe.
df = pd.read_excel('XOR_dataset.xlsx', 'Sheet1')

# Numpy array - https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array-preserving-index
data = df.values
ans_set = data[:, 0:1]  # first column; Y
inp_set = data[:, 1:]  # input matrix; X

# Defining a learning rate
eta = 0.5

# Creating a neural network object.
nn = NeuralNetwork(2, 4, 1)

# Training the neural network.
#nn.train(data, 15000, eta)
nn.train(ans_set, inp_set, 15000, eta)

# Validating the neural network.
print('Success rate: ', nn.validate(data))

# Let's compare this to the Keras neural network libraries. We'll learn that and also do validation on our own code at the same time.

#model = Sequential([
#    Dense(2, input_shape=(2,)),
#    Activation('sigmoid'),
#    Dense(1),
#    Activation('sigmoid'),
#])

#model.compile(loss='mean_squared_error', optimizer='rmsprop')
#y = data[:, 0]  # first column
#X = data[:, 1:]
#model.fit(X, y, batch_size=1, epochs=10000)
#print(model.predict_proba(X))