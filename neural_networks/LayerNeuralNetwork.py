__author__ = 'sebastian'

# source: http://iamtrask.github.io/2015/07/12/basic-python-network/

import numpy as np


# maps any value to value between 0, 1
def sigmoid(x, deriv=False):
    if deriv:
        return sigmoid(x)*(1-sigmoid(x))
    return 1 / (1 + np.exp(-x))

# input matrix; each row is a training example
x = np.array([[0,0,1],
            [0,1,1],
            [1,0,1],
            [1,1,1]])

# output matrix; each row is a training example
y = np.array([[0],
			[1],
			[1],
			[0]])

# seed random numbers, take them from the same random distribution
np.random.seed(1)

# randomly initialize our weights with mean 0
# first layer of weights, synapse 0, connects l0 to l1
syn0 = 2*np.random.random((3,4)) - 1

# second layer of weights, synapse 1, connects l1 to l2
syn1 = 2*np.random.random((4,1)) - 1

for j in xrange(60000):

	# Feed forward through layers 0, 1, and 2
    l0 = x # first layer, specified by input data
    # x.dot(y): x, y are matrices --> matrix multiplication
    # n x m matrix dot m x p matrix --> n x p matrix
    l1 = sigmoid(np.dot(l0,syn0)) # second layer (hidden layer)
    l2 = sigmoid(np.dot(l1,syn1)) # third layer, output layer

    # amount the neural network "missed" the output
    l2_error = y - l2

    if (j% 10000) == 0:
        print "Error:" + str(np.mean(np.abs(l2_error)))

    # error of the network scaled by its confidence
    l2_delta = l2_error * sigmoid(l2,deriv=True)

    # backpropagation step; sends error across weights from l2 to l1
    # error of hidden layer, l2 delta weighted by synapse 1 weights
    l1_error = l2_delta.dot(syn1.T)

    # * = elementwise multiplication, two vectors of equal size genearte
    # an output vector of identical size
    # l1 error of the network scaled by the confidence
    l1_delta = l1_error * sigmoid(l1,deriv=True)

    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
