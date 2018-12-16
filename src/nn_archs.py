import numpy as np
from read_data import get_data
import sys
sys.path.append("network/")
from NN import NeuralNet
from metrics import gain_chart, prob_acc

#X, Y = get_data(normalized = False, standardized = False)
X, Y = get_data()

#with open("nn_arch.txt", "w+") as f:

    f.write("Activation | Hidden layers | Nodes | Area Test | R2 Test | Error rate \n")
    f.write("-"*70)

    for act in ['tanh', 'sigmoid', 'relu']:
        for size in [5, 10, 20, 50, 100]:
            for n_lay in [1, 2, 3]:
                nn = NeuralNet(X, Y.flatten(), nodes = [23] + [size]*n_lay + [2], \
                    activations = [act]*n_lay + [None], cost_func = 'log')
                nn.split_data(frac = 0.5, shuffle = True)
                nn.TrainNN(epochs = 2000, batchSize = 200, eta0 = 0.01, n_print = 100)

                ypred_test = nn.feed_forward(nn.xTest, isTraining=False)
                acc = nn.accuracy(nn.yTest, ypred_test)
                err_rate = 1 - acc/100
                area = gain_chart(nn.yTest, ypred_test, plot=False)
                R2 = prob_acc(nn.yTest, ypred_test, plot=False)

                f.write("\n  %s  |  %i  |  %i  |  %.5f  |  %.5f  |  %.3f "\
                        %(act, n_lay, size, area, R2, err_rate))
        f.write("\n" + "-"*70)
