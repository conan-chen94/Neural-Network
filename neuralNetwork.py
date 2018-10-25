# This is the main script for implementing a neural network with an input layer, a hidden layer, and an output layer.

# Importing the numpy packages so I have access to arrays, matrices, etc.
import numpy as np
import pandas as pd
import math
from scipy.special import \
	expit  # Fastest per https://stackoverflow.com/questions/43024745/applying-a-function-along-a-numpy-array

# Importing the excel data for the XOR problem.
df = pd.read_excel('XOR_dataset.xlsx', 'Sheet1')  # Pandas dataframe

# Numpy array - https://stackoverflow.com/questions/13187778/convert-pandas-dataframe-to-numpy-array-preserving-index
data = df.values


# Writing out the Neural Network class
class NeuralNetwork:
	# Constructor function
	def __init__(self, numinputs, numhidden, numoutputs):
		self.numInputs = numinputs
		self.numHidden = numhidden
		self.numOutputs = numoutputs
		self.weightsIH = np.zeros(shape=(numhidden, numinputs))
		self.weightsHO = np.zeros(shape=(numoutputs, numhidden))
		self.biasH = np.zeros(shape=(numhidden, 1))
		self.biasO = np.zeros(shape=(numoutputs, 1))

	def train(self, dataSet, epochs=5, eta=0.5):
		ansSet = dataSet[:, 0]  # first column
		trnSet = dataSet[:, 1:]
		sumSqErr = 0

		for epoch in range(epochs):

			print('Epoch number: ', epoch)
			sumSqErr = 0
			iterations = range(trnSet.shape[0])
			for i in iterations:
				x = trnSet[i, :]
				x = x.reshape(x.shape[0], 1)
				y = ansSet[i]
				z_hid, a_hid, z_out, y_hat = self.feedforward(x)
				sumSqErr = sumSqErr + (y - y_hat) ** 2
				temp_weights_ho = self.weightsHO + eta * (y - y_hat) * np.multiply(expit(z_out), 1-expit(z_out)) * np.transpose(a_hid)
				temp_weights_ih = self.weightsIH + eta * (y - y_hat) * np.multiply(expit(z_out), 1-expit(z_out)) * np.transpose(self.weightsHO) * np.multiply(expit(z_hid), 1-expit(z_hid)) * x
				temp_bias_o = self.biasO + eta * (y - y_hat) * np.multiply(expit(z_out), 1-expit(z_out))
				temp_bias_h = self.biasH + eta * (y - y_hat) * np.multiply(expit(z_out), 1-expit(z_out)) * np.multiply(np.transpose(self.weightsHO), np.multiply(expit(z_hid), 1-expit(z_hid)))

				# Fourth, updating everything.
				self.weightsHO = temp_weights_ho
				self.weightsIH = temp_weights_ih
				self.biasO = temp_bias_o
				self.biasH = temp_bias_h
			print('Sum Squared Error:', np.asscalar(sumSqErr))

			if sumSqErr < 0.001:
				break


	def validate(self,dataSet):
		ansSet = dataSet[:, 0]  # first column
		valSet = dataSet[:, 1:]

		totalCorrect = 0
		totalTested = valSet.shape[0]
		iterations = range(valSet.shape[0])

		for i in iterations:
			x = valSet[i, :]
			x = x.reshape(x.shape[0], 1)
			y = ansSet[i]
			z_hid, a_hid, z_out, y_hat = self.feedforward(x)
			if round(np.asscalar(y_hat)) == y:
				totalCorrect += 1

		return totalCorrect/totalTested


	def feedforward(self, x):
		z_hid = np.matmul(self.weightsIH, x) + self.biasH
		a_hid = expit(z_hid)
		z_out = np.matmul(self.weightsHO, a_hid) + self.biasO
		y_hat = expit(z_out)
		return z_hid, a_hid, z_out, y_hat

	@staticmethod
	def sigmoid(x):
		return 1 / (1 + math.exp(-x))


# Read in data for the XOR Dataset. Alternatively, we can just hard-code this in.


# Defining a learning rate
eta = 0.5

# Creating a neural network object.
nn = NeuralNetwork(2, 2, 1)

# Training the neural network.
nn.train(data, 15000, eta)

# Validating the neural network.
print('Success rate: ', nn.validate(data))

