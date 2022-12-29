import numpy as np
import time

class ReLUNetwork:
    def __init__(self, sizes, epochs, learnRate):
        self.sizes = sizes
        self.epochs = epochs
        self.learnRate = learnRate

        # Network structure
        inputLayer = self.sizes[0]
        hiddenOne = self.sizes[1]
        hiddenTwo = self.sizes[2]
        outputLayer = self.sizes[3]

        # Weights
        self.parameters = {
            'w1': np.random.randn(hiddenOne, inputLayer) * np.sqrt(1. / hiddenOne),
            'w2': np.random.randn(hiddenTwo, hiddenOne) * np.sqrt(1. / hiddenTwo),
            'w3': np.random.randn(outputLayer, hiddenTwo) * np.sqrt(1. / outputLayer),
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

    def forwardPass(self, input):
        parameters = self.parameters

        # inputLayer
        parameters['a0'] = input

        # hiddenOne
        parameters['z1'] = np.dot(parameters['w1'], parameters['a0'])
        parameters['a1'] = self.ReLULayer(parameters['z1'])

        # hiddenTwo
        parameters['z2'] = np.dot(parameters['w2'], parameters['a1'])
        parameters['a2'] = self.ReLULayer(parameters['z2'])

        # outputLayer
        parameters['z3'] = np.dot(parameters['w3'], parameters['a2'])
        parameters['a3'] = self.ReLULayer(parameters['z3'])

        return parameters['a3']

    def backwardPass(self, input, output):
        parameters = self.parameters
        updates = {}

        # w3 update
        updater = 2 * (output - input) / output.shape[0] * self.SoftmaxLayer(parameters['z3'], derivative=True)
        updates['w3'] = np.outer(updater, parameters['a2'])

        # w2 update
        updater = np.dot(parameters['w3'].T, updater) * self.SoftmaxLayer(parameters['z2'], derivative=True)
        updates['w2'] = np.outer(updater, parameters['a1'])

        # w1 update
        updater = np.dot(parameters['w2'].T, updater) * self.SoftmaxLayer(parameters['z1'], derivative=True)
        updates['w1'] = np.outer(updater, parameters['a0'])

        return updates

    # Optimizer that controls the weight
    def optimizer(self, updates):
        for key, value in updates.items():
            self.parameters[key] -= self.learnRate * value

    def accuracyCalc(self, data, workers):
        accuracies = []

        for i in data:
            values = i.split(',')
            input = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
            target = np.zeros(workers) + 0.01
            target[int(values[0])] = 0.99
            output = self.forwardPass(input)
            accuracy = np.argmax(output)
            accuracies.append(accuracy == np.argmax(target))

        return np.mean(accuracies)

    def train(self, data, test, workers):
        start = time.time()

        print("ReLU Network: ")
        for i in range(self.epochs):
            for j in data:
                values = j.split(',')
                input = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
                target = np.zeros(workers) + 0.01
                target[int(values[0])] = 0.99
                output = self.forwardPass(input)

                updates = self.backwardPass(target, output)
                self.optimizer(updates)

            trainAccuracy = self.accuracyCalc(data, workers)
            testAccuracy = self.accuracyCalc(test, workers)

            print("Epoch", i + 1)
            print("Time =", time.time() - start)
            print("Train Accuracy =", trainAccuracy * 100)
            print("Test Accuracy =", testAccuracy * 100)