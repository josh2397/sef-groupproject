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

        self.weights = [np.random.normal(0, 1/np.sqrt(shape[i+1], (shape[i], shape[i+1])) for i in range(self.numberOfLayers - 1)))]

        self.bias = [np.random.normal(0, 1, (shape[i])) for i in range(1, self.numberOfLayers)]


network = convolutionalNeuralNetwork( [784, 30, 10] )