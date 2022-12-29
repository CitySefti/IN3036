import numpy as np
import time

class ReLUNetwork:
    def init(self, sizes, epochs, learnRate):
        self.sizes = sizes
        self.epochs = epochs
        self.learnRate = learnRate

        inputLayer = self.sizes[0]
        hiddenOne = self.sizes[1]
        hiddenTwo = self.sizes[2]
        outputLayer = self.sizes[3]

        self.params = {
            'w1': np.randomrandn(hiddenOne, inputLayer) * np.sqrt(1. / hiddenOne),
            'w2': np.randomrandn(hiddenTwo, hiddenOne) * np.sqrt(1. / hiddenTwo),
            'w3': np.randomrandn(outputLayer, hiddenTwo) * np.sqrt(1. / outputLayer),
        }

    def ReLULayer(self, input, derivative=False):
        if derivative:
            return 1 * (input > 0) # Backward
        return np.maximum(0, input)  # Forward

    def SoftmaxLayer(self, input, derivative=False):
        exponents = np.exp(input - np.max(input))
        if derivative:
            return exponents / np.sum(exponents) * (1 - exponents / np.sum(exponents))  # Backward
        return exponents / np.sum(exponents) # Forward

    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, gradientOutput):
        gradientInput = gradientOutput
        for layer in reversed(self.layers):
            gradientInput = layer.backward(gradientInput)
        return gradientInput

    def train(self, input, target, learningRate):
        # Forward pass
        output = self.forward(input)
        # Calculate the loss
        loss = self.lossFunction(output, target)
        # Calculate the gradient of the loss with respect to the output
        gradientOutput = self.lossFunction.gradient(output, target)
        # Backward pass
        self.backward(gradientOutput)
        # Update the weights and biases of the layers
        for layer in self.layers:
            if hasattr(layer, 'weights'):
                layer.weights -= learningRate * layer.gradientWeights
            if hasattr(layer, 'biases'):
                layer.biases -= learningRate * layer.gradientBiases
        return loss

    def evaluate(self, input, target):
        output = self.forward(input)
        loss = self.lossFunction()