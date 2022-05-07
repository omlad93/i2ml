"""
Omri Elad 204620702
Python 3.10
backprop_network.py
"""
import random
import numpy as np
import math


class Network(object):

    def __init__(self, sizes):
        """
        The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def reset(self):
        '''
        reset Network's biases & weights
        '''
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]


    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data,quick=False, continuos=False,silent=False):
        """
        Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  
        quick = True -> no calculation of training accuracy & loss
        """     
        if not continuos:
            self.reset()
        test_accuracy,train_accuracy,train_loss = [],[],[]
        print(f"\t\t{learning_rate=},{mini_batch_size=}")
        if not silent:
            print("\t\t\tInitial Test Accuracy: {0}".format(self.one_label_accuracy(test_data)))
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)
                ]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if not silent:
                print("\t\t\tEpoch {0} Test Accuracy: {1}".format(j, self.one_label_accuracy(test_data)))
            
            test_accuracy.append(self.one_label_accuracy(test_data))
            assert test_accuracy[-1] >= 0 and  test_accuracy[-1] <=1
            if not quick:
                train_accuracy.append(self.one_hot_accuracy(training_data))
                assert train_accuracy[-1] >= 0 and  train_accuracy[-1] <=1
                train_loss.append(self.loss(training_data))
        if silent:
            print(f"\t\t\tFinal Test Accuracy: {test_accuracy[-1]}")

        return train_accuracy,train_loss,test_accuracy

    def update_mini_batch(self, mini_batch, learning_rate):
        """
        Update the network's weights and biases by applying
        stochastic gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``.
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    def forward(self, x: np.ndarray):
        '''
        Calculates Vs:(v=w*z+b) and returns dVs/dReLU and Zs:(relu(v))
        '''
        z = x.copy()
        z_list, dv_list = [z], []
        for w, b in zip(self.weights, self.biases):
            v = np.dot(w, z)+b
            dv_list.append(relu_derivative(v))
            z = relu(v)
            z_list.append(z)
        return z_list, dv_list

    def backward(self, z_list, dv_list, y):
        '''
        calculate and returns Deltas
        '''
        d_list = [ None for i in range(len(self.weights))]
        d_list[-1] = self.loss_derivative_wr_output_activations(z_list[-1], y)
        for i in reversed(range(len(self.weights) - 1)):
            w_delta = np.dot(np.transpose(self.weights[i+1]),d_list[i+1])
            d_list[i] = w_delta * relu_derivative(dv_list[i])
        return d_list

    def backprop(self, x, y):
        """
        The function receives as input a 784 dimensional 
        vector x and a one-hot vector y.
        The function should return a tuple of two lists (db, dw) 
        as described in the assignment pdf. 
        """
        z_list, dv_list = self.forward(x)
        db = self.backward(z_list, dv_list, y)
        dw = [None for _ in range(len(self.weights))]
        for i,(delta,z) in enumerate(zip(db,z_list)):
            dw[i] = np.dot(delta, np.transpose(z))
        return db,dw

    def one_label_accuracy(self, data):
        """Return accuracy of network on data with numeric labels"""
        output_results = [
            (np.argmax(self.network_output_before_softmax(x)), y)
            for (x, y) in data
        ]
        return sum(int(x == y) for (x, y) in output_results)/float(len(data))

    def one_hot_accuracy(self, data):
        """Return accuracy of network on data with one-hot labels"""
        output_results = [
            (np.argmax(self.network_output_before_softmax(x)), np.argmax(y))
            for (x, y) in data
        ]
        return sum(int(x == y) for (x, y) in output_results) / float(len(data))

    def network_output_before_softmax(self, x):
        """Return the output of the network before softmax if ``x`` is input."""
        layer = 0
        for b, w in zip(self.biases, self.weights):
            if layer == len(self.weights) - 1:
                x = np.dot(w, x) + b
            else:
                x = relu(np.dot(w, x)+b)
            layer += 1
        if any(np.isnan(x)):
            stop= True
        return x

    def loss(self, data):
        """Return the loss of the network on the data"""
        loss_list = []
        for (x, y) in data:
            net_output_before_softmax = self.network_output_before_softmax(x)
            net_output_after_softmax = self.output_softmax(net_output_before_softmax)
            loss_list.append(np.dot(-np.log(net_output_after_softmax).transpose(), y).flatten()[0])
        return sum(loss_list) / float(len(data))

    def output_softmax(self, output_activations):
        """Return output after softmax given output before softmax"""
        output_exp = np.exp(output_activations - np.max(output_activations))
        if any(np.isnan(output_exp/np.sum(output_exp))):
            stop = True
        return output_exp/output_exp.sum()

    def loss_derivative_wr_output_activations(self, output_activations, y):
        '''
        Derivative of loss with respect to the output activations before softmax
        '''
        return self.output_softmax(output_activations)-y


def relu(z):
    """
    ReLU function.
    """
    relu = np.copy(z)
    relu[relu<=0] = 0
    return relu


def relu_derivative(z):
    """
    d(ReLU(z))
    ---------
        dz
    """
    return 1.0*(z>0)
