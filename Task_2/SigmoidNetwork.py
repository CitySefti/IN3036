import numpy as np
import matplotlib.pyplot as plt
import time


class SigmoidNetwork:
    def __init__(self, sizes, epochs, learnRate, dropRates):
        self.sizes = sizes
        self.epochs = epochs
        self.learnRate = learnRate
        self.dropRates = dropRates

        # Network structure, giving each layer a number of nodes
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

    def sigmoidLayer(self, input, derivative=False):
        if derivative:
            return (np.exp(-input)) / ((np.exp(-input) + 1) ** 2)  # Backward
        return 1 / (1 + np.exp(-input))  # Forward

    def softmaxLayer(self, input, derivative=False):
        exponents = np.exp(input - np.max(input))
        if derivative:
            return exponents / np.sum(exponents) * (1 - exponents / np.sum(exponents))  # Backward
        return exponents / np.sum(exponents)  # Forward

    def dropOut(self, layer, dropRate):
        randMatrix = np.random.randn(*layer.shape)
        layer = layer * (randMatrix > dropRate)
        layer = layer / (1 - dropRate)
        return layer

    def forwardPass(self, input):
        parameters = self.parameters

        # -> inputLayer activation
        parameters['a0'] = input

        # -> hiddenOne activation
        parameters['z1'] = np.dot(parameters['w1'], parameters['a0'])
        parameters['a1'] = self.sigmoidLayer(parameters['z1'])
        if self.dropRate > 0:
            parameters['a1'] = self.dropOut(parameters['a1'], self.dropRates[0])

        # -> hiddenTwo activation
        parameters['z2'] = np.dot(parameters['w2'], parameters['a1'])
        parameters['a2'] = self.sigmoidLayer(parameters['z2'])
        if self.dropRate > 0:
            parameters['a2'] = self.dropOut(parameters['a2'], self.dropRates[1])

        # -> outputLayer activation
        parameters['z3'] = np.dot(parameters['w3'], parameters['a2'])
        parameters['a3'] = self.sigmoidLayer(parameters['z3'])

        return parameters['a3']

    def backwardPass(self, input, output):
        parameters = self.parameters
        updates = {}

        # w3 update
        updater = 2 * (output - input) / output.shape[0] * self.softmaxLayer(parameters['z3'], derivative=True)
        updates['w3'] = np.outer(updater, parameters['a2'])

        # w2 update
        updater = np.dot(parameters['w3'].T, updater) * self.softmaxLayer(parameters['z2'], derivative=True)
        updates['w2'] = np.outer(updater, parameters['a1'])

        # w1 update
        updater = np.dot(parameters['w2'].T, updater) * self.softmaxLayer(parameters['z1'], derivative=True)
        updates['w1'] = np.outer(updater, parameters['a0'])

        return updates

    # Optimizer that controls the weight
    def optimizer(self, updates):
        for key, value in updates.items():
            self.parameters[key] -= self.learnRate * value

    # This is for calculating accuracy
    def accuracyCalc(self, test, workers):
        accuracies = []

        for i in test:
            values = i.split(',')
            input = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
            target = np.zeros(workers) + 0.01
            target[int(values[0])] = 0.99
            output = self.forwardPass(input)
            accuracy = np.argmax(output)
            accuracies.append(accuracy == np.argmax(target))

        return np.mean(accuracies)

    def train(self, train, test, workers):
        accuracies = []

        start = time.time()
        print("Sigmoid Network: ")
        for i in range(self.epochs):
            print("Epoch " + str(i + 1))
            for j in train:
                values = j.split(',')
                input = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
                target = np.zeros(workers) + 0.01
                target[int(values[0])] = 0.99
                output = self.forwardPass(input)

                updates = self.backwardPass(target, output)
                self.optimizer(updates)

            testAccuracy = self.accuracyCalc(test, workers)
            accuracies.append(testAccuracy)

            print("Time = " + str(time.time() - start) + " Accuracy = " + str(testAccuracy * 100))

        plt.subplots(figsize=(10, 10))
        plt.plot(range(self.epochs), accuracies)
        plt.show()
