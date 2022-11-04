
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
            Input from the (output of) previous layer
            (the dimension must be equal to the fan_in).

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
            Input from the (output of) previous layer
            (the dimension must be equal to the fan_in).

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

def sigmoid(net):
    """Activation function for classification
    purposes with output range [0, 1].

    Arguments
    ----------
    net : float
        Computation rsult of linera combination of inputs
        with the weights of the units.
    """
    return 1. / (1 +  np.exp(-net))

def sigmoid_prime(net):
    """Activation function derivative.

    Arguments
    ----------
    net : float
        Computation rsult of linera combination of inputs
        with the weights of the units.
    """
    return sigmoid(net) * (1 - sigmoid(net))

if __name__ == "__main__":
    # Checking the output unit.
    TARGET = 0.9
    ETA = 0.2
    myoutunit = OutputUnit(5, sigmoid, sigmoid_prime)
    print(f'Weights at the initailization: {myoutunit.weights}')
    print(f'A value for activation function: {myoutunit.activation(-1)}')
    print(f'A value for activation function: {myoutunit.activation(0)}')
    print(f'A value for activation function: {myoutunit.activation(1)}')
    print(f'A value for activation function derivative: {myoutunit.derivative(-1)}')
    print(f'A value for activation function derivative: {myoutunit.derivative(0)}')
    print(f'A value for activation function derivative: {myoutunit.derivative(1)}')
    input_array = np.array([1, 2, 3, 4, 5])
    computation = myoutunit.forward_propagation(input_array)
    print(f'Result for the computation of the unit: {computation}')
    error = TARGET - computation
    delta_unit = myoutunit.backward_propagation(error, ETA)
    new_weights = myoutunit.weights
    print(f'Computed delta for weight correction: {delta_unit}')
    print(f'Weights aftr training: {new_weights} \n')
    # checking the hidden unit.
    myhiddenunit = HiddenUnit(5, sigmoid, sigmoid_prime)
    print(f'Weights at the initailization: {myhiddenunit.weights}')
    computation = myhiddenunit.forward_propagation(input_array)
    print(f'Result for the computation of the unit: {computation}')
    w_kj = np.random.uniform(-1, 1, 5)
    delta_k = np.random.uniform(-0.5, 0.5, 5)
    delta_unit = myhiddenunit.backward_propagation(w_kj, delta_k, ETA)
    new_weights = myhiddenunit.weights
    print(f'Computed delta for weight correction: {delta_unit}')
    print(f'Weights aftr training: {new_weights}')
