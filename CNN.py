import numpy as np

def sigmoid(x):
    return np.nan_to_num(1/(1+np.exp(-x)))

def sigmoidDerivative(x):
    return sigmoid(x)*(1-sigmoid(x))

class costFunction:
    @staticmethod
    def quadratic(summation, label):
        return 0.5*(summation - label)**2

    @staticmethod
    def quadraticDerivative(summation, label):
        return summation - label

    @staticmethod
    def delta(inputs, summation, label):
        return(summation - label)*sigmoidDerivative(inputs)

class convolutionalNeuralNetwork:

    def __init__(self, shape, cost = costFunction):
        self.shape = shape

        self.numberOfLayers = len(shape)
        '''
        setup weights for convolution architecture including max-pooling and fully connected layers
        '''
        self.weights = [np.random.normal(0, 1 / np.sqrt(shape[i + 1]), (shape[i], shape[i + 1])) for i in range(self.numberOfLayers - 1)]

        self.bias = [np.random.normal(0, 1, (shape[i])) for i in range(1, self.numberOfLayers)]

    def feedForward(self, inputdata):

        self.layerInput = []

        self.layerOutput = {}

        self.layerInput[0] = inputdata

        self.layerOutput = np.array(inputdata)

        for layer in range(1, self.numberOfLayers):
            self.layerInput[layer] += np.dot(self.layerOutput[layer - 1], self.weights[layer - 1] + self.bias[layer - 1])

            self.layerOutput[layer] = np.array(sigmoid(self.layerInput[layer]))

            return self.layerOutput[self.numberOfLayers - 1]

    def trainNetwork(self, data, miniBatchSize, numberOfEpochs, earningRate):

        #TODO - open data files

        for epoch in np.arange(numberOfEpochs):

    def storeWeightsAndBias(self):
        #store the weights and bias in their separate respective files

    def miniBatch(self):
        #separate the data into a batch and execute feedforward followed by backpropagation




network = convolutionalNeuralNetwork( [4096, 300, 250, 200, 150, 100, 50, 10] )
#call trainNetwork as main function