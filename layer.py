
"""Defining a class for a layer
of a Neural Network.
"""

import numpy as np
from neuron import OutputUnit, HiddenUnit

class OutputLayer:
    """Output layer class."""
    def __init__(self, fan_in, activation_function, activation_derivative, units_number):
        """Initialization of the layer with the number
        of required units.

        Arguments
        ----------
        fan_in : int
            Number of input lines for the unit.

        activation_function : funtion
            Function apllied to net for the computation of the output.

        activation_derivative : function
            Derivative of the activation function.

        units_number : int
            Number of neurons in the layer.
        """
        self.input = None
        self.output = None
        self.deltas = None
        self.matrix = None
        self.fan_in = fan_in
        self.number = units_number
        self.activation = activation_function
        self.derivative = activation_derivative
        self.units = np.empty(shape=0)
        for _ in range(units_number):
            self.units = np.append(self.units, \
            OutputUnit(self.fan_in, self.activation, self.derivative))

    def forward_propagation(self, input_data):
        """Method for the (units of the) layer computation.

        Arguments
        ----------
        input_data : float or arraylaike
            Result from the previous layer.

        Return
        ----------
        self.oyutput : float or arraylike
            Result from the present layer.
        """
        self.input = input_data
        self.output = np.empty(shape=0)
        for neuron in self.units:
            self.output = np.append(self.output, neuron.forward_propagation(self.input))
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        """Method for the (units in the layer) weights adjustment.

         Arguments
        ----------
        output_error : float
            Difference between unit output and target.

        learning rate : float
            Parameter for the speed control of the algorithm.

        Return
        ---------
        deltas : float or arraylike
            Deltas for the following backprop step.
        """
        self.deltas = np.empty(shape=0)
        for neuron in self.units:
            self.deltas = np.append(self.deltas, \
            neuron.backward_propagation(self.input, output_error, learning_rate))
        return self.deltas

    def weights_matrix(self):
        """Method to get all the weights of all the unit
        in the layer in form of a matrix (each row is
        comosed by weights from a single unit)
        """
        self.matrix = np.empty(shape=(0,0))
        for neuron in self.units:
            self.matrix = np.r_[self.matrix, neuron.weights]
        return self.matrix

class HiddenLayer:
    """Output layer class."""
    def __init__(self, fan_in, activation_function, activation_derivative, units_number):
        """Initialization of the layer with the number
        of required units.

        Arguments
        ----------
        fan_in : int
            Number of input lines for the unit.

        activation_function : funtion
            Function apllied to net for the computation of the output.

        activation_derivative : function
            Derivative of the activation function.

        units_number : int
            Number of neurons in the layer.
        """
        self.input = None
        self.output = None
        self.deltas = None
        self.matrix = None
        self.fan_in = fan_in
        self.number = units_number
        self.activation = activation_function
        self.derivative = activation_derivative
        self.units = np.empty(shape=0)
        for _ in range(units_number):
            self.units = np.append(self.units, \
            HiddenUnit(self.fan_in, self.activation, self.derivative))

    def forward_propagation(self, input_data):
        """Method for the (units of the) layer computation.

        Arguments
        ----------
        input_data : float or arraylaike
            Result from the previous layer.

        Return
        ----------
        self.oyutput : float or arraylike
            Result from the present layer.
        """
        self.input = input_data
        self.output = np.empty(shape=0)
        for neuron in self.units:
            self.output = np.append(self.output, neuron.forward_propagation(self.input))
        return self.output

    def backward_propagation(self, weight_next, delta_next, learning_rate):
        """Method for the (units in the layer) weights adjustment.

         Arguments
        ----------
        output_error : float
            Difference between unit output and target.

        learning rate : float
            Parameter for the speed control of the algorithm.

        Return
        ---------
        deltas : float or arraylike
            Deltas for the following backprop step.
        """
        self.deltas = np.empty(shape=0)
        index = 0
        for neuron in self.units:
            self.deltas = np.append(self.deltas, \
            neuron.backward_propagation(self.input, weight_next[:, index], delta_next, learning_rate))
            index += 1
        return self.deltas

    def weights_matrix(self):
        """Method to get all the weights of all the unit
        in the layer in form of a matrix (each row is
        comosed by weights from a single unit)
        """
        self.matrix = np.empty(shape=(0,0))
        for neuron in self.units:
            self.matrix = np.r_[self.matrix, neuron.weights]
        return self.matrix
