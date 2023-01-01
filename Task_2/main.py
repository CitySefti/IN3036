from SigmoidNetwork import SigmoidNetwork
from ReLUNetwork import ReLUNetwork

trainFile = open("data/mnist_train.csv")
trainData = trainFile.readlines()
trainFile.close()

testFile = open("data/mnist_test.csv")
testData = testFile.readlines()
testFile.close()

sig = SigmoidNetwork(sizes=[784, 128, 64, 10], epochs=10, learnRate=0.1, dropRates=[0.2, 0.2])
sig.train(trainData, testData, 10)

relu = ReLUNetwork(sizes=[784, 128, 64, 10], epochs=10, learnRate=0.1, dropRates=[0.2, 0.2])
relu.train(trainData, testData, 10)

"""
The sigmoid function can be put in place of softmax in backward pass. 
Just make sure derivative is set to true when calling.
The relu function can not be used in backward pass due to a run-time warning. 

Softmax can also be put in forward pass, just don't set derivative to true.

"""
