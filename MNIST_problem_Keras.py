# Importing the Keras module. https://keras.io/.
from keras.models import Sequential
from keras.layers import Dense, Convolution2D, MaxPooling2D, Conv2D, Flatten
from keras import optimizers
import keras

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

# Preprocessing to have our y's be in the form of a vector, such as [0 0 0 1 0 0 0 0 0 0] to represent '4'.
MNIST_train_ans_set_vector = MNIST_train[:, 0:1]  # first column; Y
MNIST_train_ans_set = np.zeros(shape=(MNIST_train_ans_set_vector.shape[0], 10))
for i in range(MNIST_train_ans_set_vector.shape[0]):
    loc = MNIST_train_ans_set_vector[i, 0]  # the image label
    MNIST_train_ans_set[i, loc] = 1

# Preprocessing to have our x's be 2D tensors with elements of 0's and 1's only, instead of elements in [0,255].
MNIST_train_inp_set_vector = MNIST_train[:, 1:] / 255
MNIST_train_inp_set_vector_ceil = np.ceil(MNIST_train_inp_set_vector)  # input matrix; X
MNIST_train_inp_set_matrix_ceil = MNIST_train_inp_set_vector_ceil.reshape(42000, 28, 28, 1)

# Firstly following this tutoring on Keras. http://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
# Creating the network model.
model = Sequential()
model.add(Convolution2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compiling the model.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

# Training the model.
x_train = MNIST_train_inp_set_matrix_ceil
y_train = MNIST_train_ans_set
batch_size = 100
epochs = 10

# Defining an AccuracyHistory class to use the callback parameter.
# class AccuracyHistory(keras.callbacks.Callback):
#    def on_train_begin(self, logs={}):
#        self.acc = []
#
#    def on_epoch_end(self, batch, logs={}):
#        self.acc.append(logs.get('acc'))
#
#history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_train, y_train))


# Creating a neural network object.
# MNIST_nn = NeuralNetwork(784, 50, 10)

# Training the neural network.
# MNIST_nn.train(MNIST_train_ans_set, MNIST_train_inp_set, 50, eta, .05)

# Using the model to evaluate the training set one more time.
score = model.evaluate(x_train, y_train, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# Using the trained neural network to generate predictions on the test set.
MNIST_test_df = pd.read_csv('MNIST_test.csv')
MNIST_test = MNIST_test_df.values  # no labels in the test set
MNIST_test_inp_set_vector = MNIST_test / 255  # input matrix; X
MNIST_test_inp_set_vector_ceil = np.ceil(MNIST_test_inp_set_vector)
MNIST_test_inp_set_matrix_ceil = MNIST_test_inp_set_vector_ceil.reshape(42000, 28, 28, 1)

# MNIST_test_inp_set_transpose = np.transpose(MNIST_test_inp_set)

# Loop through and feed forward every image to generate the prediction.
x_test = MNIST_test_inp_set
y_hat_test = model.predict(x_test, batch_size=100, verbose=1)

# Translating the y_test predictions into a one-number label for each image.
MNIST_results = np.empty(shape=(MNIST_test_inp_set.shape[0], 2))
for i in range(MNIST_test_inp_set.shape[0]):
    y_hat_prediction = np.where(y_hat_test == np.amax(y_hat_test))
    MNIST_results[i, 0] = i+1
    MNIST_results[i, 1] = y_hat_test[0]

MNIST_results_df = pd.DataFrame(data=MNIST_results, columns=['ImageID', 'Label'])
MNIST_results_df.to_csv('MNIST_submission.csv', index=False)

# Printing the execution time
MNIST_end = time.time()
print('\n', 'Execution time (s): ', format(MNIST_end-MNIST_start, '.10f'), '\n')