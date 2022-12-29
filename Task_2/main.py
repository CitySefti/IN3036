from SigmoidNetwork import SigmoidNetwork

trainFile = open("data/mnist_train.csv")
trainData = trainFile.readlines()
trainFile.close()

testFile = open("data/mnist_test.csv")
testData = testFile.readlines()
testFile.close()


sig = SigmoidNetwork(sizes=[784, 128, 64, 10], epochs=10, learnRate=0.5)

sig.train(trainData, testData, 10)


