import numpy as np
from read_data import get_data
import sys
sys.path.append("network/")
from NN import NeuralNet
from metrics import gain_chart, prob_acc
from network.resampling import Resample as rs 
from imblearn.over_sampling import SMOTE


def initalized_neural(X: np.ndarray, Y: np.ndarray):
    nn = NeuralNet(X, Y.flatten(), nodes = [23, 10, 10,2], activations = ['tanh','tanh',None], cost_func = 'log')
    nn.split_data(frac = 0.5, shuffle = True)
    return nn

def train_neural(nn: NeuralNet): 
    nn.TrainNN(epochs = 20, batchSize = 200, eta0 = 0.01, n_print = 100)
    ypred_test = nn.feed_forward(nn.xTest, isTraining=False)
    
    print('accuracy: ', nn.accuracy(nn.yTest, ypred_test))
    gain_chart(nn.yTest, ypred_test)
    prob_acc(nn.yTest, ypred_test)


def undersample(nn: NeuralNet):
    resample = rs(nn.xTrain, nn.yTrain)
    nn.xTrain, nn.yTrain = resample.Under()
    return nn

def oversample(nn: NeuralNet):
    resample = rs(nn.xTrain, nn.yTrain)
    nn.xTrain, nn.yTrain = resample.Over()
    return nn 

def smote(nn: NeuralNet):
    sm = SMOTE()
    nn.xTrain, nn.yTrain = sm.fit_resample(nn.xTrain, nn.yTrain)
    return nn


if __name__ == '__main__':
    print('Importing data')
    X, Y = get_data(normalized = False, standardized=True)
    print('Data imported')
    nn = initalized_neural(X, Y)
    print(type(nn))
    nn_under = undersample(nn)
    nn_over = oversample(nn)
    nn_smote = smote(nn)
    train_neural(nn_under)
    train_neural(nn_over)
    train_neural(nn_smote)
