import numpy as np
import time

trainFile = open("data/mnist_train.csv")
trainData = trainFile.readlines()
trainFile.close()

testFile = open("data/mnist_test.csv")
testData = testFile.readlines()
testFile.close()

class SigmoidNetwork:
    def __init__(self, sizes, epochs, learnRate):
        self.sizes = sizes
        self.epochs = epochs
        self.learnRate = learnRate

        inputLayer = self.sizes[0]
        hiddenOne = self.sizes[1]
        hiddenTwo = self.sizes[2]
        outputLayer = self.sizes[3]

        self.parameters = {
            's1':np.random.randn(hiddenOne, inputLayer) * np.sqrt(1. / hiddenOne),
            's2':np.random.randn(hiddenTwo, hiddenOne) * np.sqrt(1. / hiddenTwo),
            's3':np.random.randn(outputLayer, hiddenTwo) * np.sqrt(1. / outputLayer),
        }

    def SigmoidLayer(self, input, derivative=False):
        if derivative:
            return (np.exp(-input)) / ((np.exp(-input) + 1) ** 2) # Backward
        return 1 / (1 + np.exp(-input)) # Forward

    def SoftmaxLayer(self, input, derivative=False):
        exponents = np.exp(input - np.max(input))
        if derivative:
            return exponents / np.sum(exponents) * (1 - exponents / np.sum(exponents)) # Backward
        return exponents / np.sum(exponents) # Forward

    def forwardPass(self, input):
        parameters = self.parameters

        # inputLayer
        parameters['a0'] = input

        # hiddenOne
        parameters['f1'] = np.dot(parameters['s1'], parameters['a0'])
        parameters['a1'] = self.SigmoidLayer(parameters['f1'])

        # hiddenTwo
        parameters['f2'] = np.dot(parameters['s2'], parameters['a1'])
        parameters['a2'] = self.SigmoidLayer(parameters['f2'])

        # outputLayer
        parameters['f3'] = np.dot(parameters['s3'], parameters['a2'])
        parameters['a3'] = self.SigmoidLayer(parameters['f3'])

        return parameters['a3']

    def backwardPass(self, input, output):
        parameters = self.parameters
        updates = {}

        # s3 update
        updater = 2 * (output - input) / output.shape[0] * self.SoftmaxLayer(parameters['f3'], derivative=True)
        updates['s3'] = np.outer(updater, parameters['a2'])

        # s2 update
        updater = np.dot(parameters['s3'].T, updater) * self.SoftmaxLayer(parameters['f2'], derivative=True)
        updates['s2'] = np.outer(updater, parameters['a1'])

        # s1 update
        updater = np.dot(parameters['s2'].T, updater) * self.SoftmaxLayer(parameters['f1'], derivative=True)
        updates['s1'] = np.outer(updater, parameters['a0'])

        return updates

    def updateNetwork(self, updates):
        for key, value in updates.items():
            self.parameters[key] -= self.learnRate * value

    def accuracyCalc(self, workers):
        accuracies = []

        for i in trainData:
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

        print("Sigmoid Network: ")
        for i in range(self.epochs):
            for j in data:
                values = j.split(',')
                input = (np.asfarray(values[1:]) / 255.0 * 0.99) + 0.01
                target = np.zeros(workers) + 0.01
                target[int(values[0])] = 0.99
                output = self.forwardPass(input)

                updates = self.backwardPass(target, output)
                self.updateNetwork(updates)

            accuracy = self.accuracyCalc(workers)

            print("Epoch ", i + 1)
            print("Time = ", time.time() - start)
            print("Accuracy = ", accuracy * 100)

    def evaluate(self, input, target):
        output = self.forward(input)
        loss = self.lossFunction()