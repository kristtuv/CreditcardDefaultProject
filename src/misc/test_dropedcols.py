import numpy as np
from read_data import get_data
import sys
sys.path.append("network/")
from NN import NeuralNet

X, Y = get_data(normalized=False, standardized=True, file='droppedX6-X11.csv')
nn = NeuralNet(X, Y.flatten(), 
                nodes = [18, 50, 50,  2], 
                activations = ['tanh','tanh', None], 
                cost_func = 'log',
                regularization='l2', 
                                lamb=0.001)
                                
nn.split_data(frac = 0.5, shuffle = True, resample=True)
nn.TrainNN(epochs = 1000, batchSize =200, eta0 = 0.01, n_print = 10)
