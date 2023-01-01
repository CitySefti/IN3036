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

    # Activation Functions
    def sigmoidLayer(self, input, derivative=False):
        if derivative:
            return (np.exp(-input)) / ((np.exp(-input) + 1) ** 2)
        return 1 / (1 + np.exp(-input))

    def softmaxLayer(self, input, derivative=False):
        exponents = np.exp(input - np.max(input))
        if derivative:
            return exponents / np.sum(exponents) * (1 - exponents / np.sum(exponents))
        return exponents / np.sum(exponents)

    # Loss Function
    def crossEntropy(self, predicted, true):
        return -np.sum(true * np.log(predicted))

    # Dropout Functions
    def dropOut(self, layer, dropRate):
        randMatrix = np.random.randn(*layer.shape) # Between 1's and 0's
        layer = layer * (randMatrix < dropRate) # if value in matrix less than dropRate, set to 0
        layer = layer / (1 - dropRate) # scale layer down by drop rate
        return layer

    # Forward Pass
    def forwardPass(self, input, dropRates):
        parameters = self.parameters

        # -> inputLayer activation
        parameters['a0'] = input

        # -> hiddenOne activation
        parameters['z1'] = np.dot(parameters['w1'], parameters['a0'])
        parameters['a1'] = self.sigmoidLayer(parameters['z1'])
        if dropRates[0] > 0:
            parameters['a1'] = self.dropOut(parameters['a1'], dropRates[0])

        # -> hiddenTwo activation
        parameters['z2'] = np.dot(parameters['w2'], parameters['a1'])
        parameters['a2'] = self.sigmoidLayer(parameters['z2'])
        if dropRates[1] > 0:
            parameters['a2'] = self.dropOut(parameters['a2'], dropRates[1])

        # -> outputLayer activation
        parameters['z3'] = np.dot(parameters['w3'], parameters['a2'])
        parameters['a3'] = self.sigmoidLayer(parameters['z3'])

        return parameters['a3']

    # Backward Pass
    def backwardPass(self, input, output, dropRates):
        parameters = self.parameters
        updates = {}

        # w3 update
        updater = 2 * (output - input) / output.shape[0] * self.softmaxLayer(parameters['z3'], derivative=True)
        updates['w3'] = np.outer(updater, parameters['a2'])

        # w2 update
        updater = np.dot(parameters['w3'].T, updater) * self.softmaxLayer(parameters['z2'], derivative=True)
        if dropRates[1] > 0:
            updater = self.dropOut(updater, dropRates[1])
        updates['w2'] = np.outer(updater, parameters['a1'])

        # w1 update
        updater = np.dot(parameters['w2'].T, updater) * self.softmaxLayer(parameters['z1'], derivative=True)
        if dropRates[0] > 0:
            updater = self.dropOut(updater, dropRates[0])
        updates['w1'] = np.outer(updater, parameters['a0'])

        return updates

    # Optimizer that controls the weight (SGD)
    def optimizer(self, updates):
        for key, value in updates.items():
            self.parameters[key] = self.parameters[key] - self.learnRate * value

    # This is for calculating accuracy with test data, it has dropRates so that I can make the rate 0 for testing
    def accuracyCalc(self, test, workers, dropRates):
        accuracies = []

        for i in test:
            values = i.split(',')
            input = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
            target = np.zeros(workers) + 0.01
            target[int(values[0])] = 0.99
            output = self.forwardPass(input, dropRates)
            accuracy = np.argmax(output)
            accuracies.append(accuracy == np.argmax(target))

        return np.mean(accuracies)

    def train(self, train, test, workers):
        # For plotting
        accuracies = []
        losses = []

        # Start Timer
        start = time.time()

        print("Sigmoid Network: ")
        for i in range(self.epochs):

            # Training
            print("Epoch " + str(i + 1))
            for j in train:
                values = j.split(',')
                input = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
                target = np.zeros(workers) + 0.01
                target[int(values[0])] = 0.99

                # Forward Pass + calc loss
                output = self.forwardPass(input, self.dropRates)
                currentLoss = self.crossEntropy(output, target)

                # Backward Pass
                updates = self.backwardPass(target, output, self.dropRates)
                self.optimizer(updates)

            testAccuracy = self.accuracyCalc(test, workers, [0, 0])
            accuracies.append(testAccuracy)
            losses.append(currentLoss)

            print("Time = " + str(time.time() - start) + " Accuracy = " + str(testAccuracy * 100))

            # Stopping Criterion if the accuracy is above threshold
            if accuracies[i] > 0.95:
                break # it stops but gives issues with the plot

        plt.subplots(figsize=(6, 6))
        plt.plot(range(self.epochs), accuracies)
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

        plt.subplots(figsize=(6, 6))
        plt.plot(range(self.epochs), losses)
        plt.title('Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()


