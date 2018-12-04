import numpy as np
from read_data import get_data
import sys
sys.path.append("network/")
from NN import NeuralNet

#X, Y = get_data(normalized = False, standardized = False)
X, Y = get_data()
print(np.count_nonzero(Y == 1))

nn = NeuralNet(X, Y.flatten(), nodes = [23, 20, 2], activations = ['tanh', None], cost_func = 'log')
nn.TrainNN(epochs = 100, batchSize = 200, eta0 = 0.01, n_print = 100)
