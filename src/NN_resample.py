import numpy as np
from read_data import get_data
import sys
sys.path.append("network/")
from NN import NeuralNet
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from resampling import Resample as rs

#Import data
X, Y = get_data(normalized=False, standardized=True)

def nn_resample(Pca: bool=0,
                smote: bool=0,
                over: bool=0,
                under: bool=0,
                X: np.ndarray=X,
                Y: np.ndarray=Y) -> None:
    np.random.seed(42)
    if Pca:
        pca = PCA(n_components=14)
        X = pca.fit_transform(X)

    ##Initialize network###
    node1 = X.shape[1]
    nn = NeuralNet(X, Y.flatten(), 
                    nodes = [node1, 3, 2], 
                    activations = ['tanh', None], 
                    cost_func = 'log',
                    regularization='l2', 
                                    lamb=0.001)
                                    
    ###Split Data###
    nn.split_data(frac = 0.5, shuffle = True)


    ##Choose resampling teqnique##
    if smote:
        # sm = SMOTE(random_state=42, sampling_strategy=1.0)
        sm = SMOTE()
        nn.xTrain, nn.yTrain = sm.fit_resample(nn.xTrain, nn.yTrain)

    elif under:
        resample = rs(nn.xTrain, nn.yTrain)
        nn.xTrain, nn.yTrain = resample.Under()

    elif over:
        resample = rs(nn.xTrain, nn.yTrain)
        nn.xTrain, nn.yTrain = resample.Over()
    ##Train network##
    nn.TrainNN(epochs = 100, batchSize =200, eta0 = 0.01, n_print = 50)

if __name__ == '__main__':
    print('SMOTE')
    nn_resample(smote=1)
    print('SMOTEPCA')
    nn_resample(Pca=1, smote=1)
    print('OVER')
    nn_resample(over=1)
    print('OVERPCA')
    nn_resample(Pca=1, over=1)
    print('UNDER')
    nn_resample(under=1)
    print('UNDERPCA')
    nn_resample(Pca=1, under=1)
