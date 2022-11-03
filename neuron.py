
"""Defining a basic class for the generic units
of a Neural Netrwork.
"""

import numpy as np

class OutputUnit:
    """Generic output unit class."""
    def __init__(self, fan_in, activation_function, activation_derivative):
        """Initializing the unit.

        Arguments
        ----------
        fan_in : int
            Number of input lines for the unit.

        activation_function : funtion
            Function apllied to net for the computation of the output.

        activation_derivative : function
            Derivative of the activation function.
        """
        self.input = None
        self.output = None
        self.net = None
        self.weights = np.random.uniform(-1. / np.sqrt(fan_in), 1. / np.sqrt(fan_in), fan_in)
        self.bias = 0.
        self.activation = activation_function
        self.derivative = activation_derivative

    def forward_propagation(self, input_data):
        """Method for the network computation.

        Arguments
        ----------
        input_data : float or arraylike
            Input from the (outèput of) previous layer.

        Return
        self.output : float
            The result from the present unit.
        """
        self.input = input_data
        self.net = np.dot(self.input, self.weights)
        self.output = self.activation(self.net + self.bias)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """Method for the weights adjustment.

        Arguments
        ----------
        output_error : float
            Difference between unit output and target.

        learning rate : float
            Parameter for the speed control of the algorithm.

        Return
        ---------
        delta : float
            Delta for the following backprop step.
        """
        delta = output_error * self.derivative(self.net)
        deltaw = delta * self.input
        self.weights = self.weights + learning_rate * deltaw
        return delta

class HiddenUnit:
    """Generic hidden unit class."""
    def __init__(self, fan_in, activation_function, activation_derivative):
        """Initializing the unit.

        Arguments
        ----------
        fan_in : int
            Number of input lines for the unit.

        activation_function : funtion
            Function apllied to net for the computation of the output.

        activation_derivative : function
            Derivative of the activation function.
        """
        self.input = None
        self.output = None
        self.net = None
        self.weights = np.random.uniform(-1. / np.sqrt(fan_in), 1. / np.sqrt(fan_in), fan_in)
        self.bias = 0.
        self.activation = activation_function
        self.derivative = activation_derivative

    def forward_propagation(self, input_data):
        """Method for the network computation.
        Arguments
        ----------
        input_data : float or arraylike
            Input from the (output of) previous layer.

        Return
        self.output : float
            The result from the present unit.
        """
        self.input = input_data
        self.net = np.dot(self.input, self.weights)
        self.output = self.activation(self.net + self.bias)
        return self.output

    def backward_propagation(self, weights_next, deltas_next, learning_rate):
        """Method for the weights adjustment.

        Arguments
        ----------
        weights_next : float or arraylike
            Array of weights on the output lines of the unit.

        deltas_next : float or arraylike
            Array of the delta for eache unit in the next layer.

        learning rate : float
            Parameter for the speed control of the algorithm.

        Return
        ---------
        delta : float
            Delta for the following backprop step.
        """
        delta = np.dot(deltas_next, weights_next) * self.derivative(self.net)
        deltaw = delta * self.input
        self.weights = self.weights + learning_rate * deltaw
        return delta
