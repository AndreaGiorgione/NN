
"""Defining a class for a layer
of a Neural Network.
"""

import numpy as np
from layer import HiddenLayer, OutputLayer

class NeuralNetwork:
    """Neural Network class."""
    def __init__(self, units, output_units, activation_function, activation_derivative):
        """
        """
        self.hlayers = np.empty(shape=0)
        self.olayer = np.empty(shape=0)
        for i in range(1, len(units)):
            self.hlayers = np.append(self.hlayers, \
            HiddenLayer(units[i-1], activation_function, activation_derivative, units[i]))
        self.olayer = np.append(self.hlayers, \
        OutputLayer(units[-1], activation_function, activation_derivative, output_units))

    def forward_propagation(self, input):
        """
        """
        for layer in self.hlayers:
            input = layer.forward_propagation(input)
        output = self.olayer.forward_propagation(input)
        return output

    def backward_porpagation(self, output_error, leraning_rate):
        """
        """
        deltas = self.olayer.backward_propagation(self, output_error, leraning_rate)
        wmatrix = self.olayer.weights_matrix()
        for layer in reversed(self.hlayers):
            deltas = layer.backward_propagation(wmatrix, deltas, leraning_rate)
            wmatrix = layer.weights_matrix()

    def error(self, output, target):
        """
        """
        return np.sum((output - target)**2)
