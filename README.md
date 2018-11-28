# Neural-Network
Simple neural network implementation in Python.

This repository contains files for a simple implementation of neural networks in Python, without using machine learning libraries like Tensorflow or Keras. I wrote this as an exercise to understand the mathematics of basic neural networks, and would recommend this path to anyone else looking for the same.

neuralNetwork.py contains the code for the NeuralNetwork class. From it, you can create instances of three-layer neural networks (one input layer, one hidden layer, and one output layer). These neural network instances can then be trained using the train() method, tested on datasets using the validate() method, and used to generate predictions with the feedforward() method.

XOR_problem.py contains the code for using the NeuralNetwork class in solving the XOR problem. XOR_dataset.xlsx contains the XOR table, which is read into a Numpy array using the Pandas library. XOR_dataset_2.xlsx also contains the XOR table, but was a test case in having a neural network architecture with more than one output.

MNIST_problem.py contains the code for using the NeuralNetwork class to learn digit recognition with the famous MNIST dataset. This was an exercise following the Kaggle tutorial located here: https://www.kaggle.com/c/digit-recognizer. MNIST_train.csv contains 48000 lines of labeled data, each line representing a 28x28 pixel image of a handwritten digit (0, 1, 2, ... or 9) along with a corresponding label. MNIST_test.csv contains 48000 lines of unlabeled data, for which the NeuralNetwork class is used to generate classification predictions. MNIST_submission.csv and MNIST_sample_submission.csv are files involved in the Kaggle submission process.
