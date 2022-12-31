from SigmoidNetwork import SigmoidNetwork
from ReLUNetwork import ReLUNetwork

trainFile = open("data/mnist_train.csv")
trainData = trainFile.readlines()
trainFile.close()

testFile = open("data/mnist_test.csv")
testData = testFile.readlines()
testFile.close()

sig = SigmoidNetwork(sizes=[784, 128, 64, 10], epochs=3, learnRate=0.1, dropRate=[0.8, 0.8])
sig.train(trainData, testData, 10)

#relu = ReLUNetwork(sizes=[784, 128, 64, 10], epochs=10, learnRate=0.1)
#relu.train(trainData, testData, 10)


