# Importing the NeuralNetwork class.
from neuralNetwork import NeuralNetwork

# Importing the pandas package so I can read in excel files.
import pandas as pd

# Importing time to time execution - https://pythonhow.com/measure-execution-time-python-code/
import time


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
print('\nValidation Success rate: ', nn.validate(ans_set, inp_set))

# Printing the execution time
end = time.time()
print('Execution time (s): ', format(end-start, '.10f'))