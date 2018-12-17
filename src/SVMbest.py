import pandas as pd
import numpy as np
import re
import time
import sys
import argparse

from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split, KFold

from network.SVM import predict
from network.SVM import _convert_string_float as csf
from read_data import get_data
from network.NN import NeuralNet
from network.resampling import Resample as rs
from imblearn.over_sampling import SMOTE

def get_values(string: str):
    """
    Takes a string with the shape of G:%s_C:%s_K:%s_D:%s_M:%s
    and returns the %s values either as float or string
    param: string: String with shape G:%s_C:%s_K:%s_D:%s_M:%s
    type: string: str
    """
    string = string.replace('_', '')
    return list(map(csf, re.sub(r'\w:', ' ', string).split()))


def make_instance(string: str, maxiter: int):
    """
    Makes svm instance
    param: string: String with shape G:%s_C:%s_K:%s_D:%s_M:%s
    type: string: str
    param: maxiter: maximum iterations in svm
    type: maxiter: int
    """
    l = get_values(string)
    g = l[0]
    c = l[1]
    k = l[2]
    d = l[3]
    m = maxiter
    return svm.SVC(kernel=k, C=c, gamma=g, degree=d, probability=True, max_iter=m)

def make_svm_dict(N_best: int, maxiter:int=50000):
    """
    Returns a dictionary of svm instances from the Area_R2.csv file
    N_best is how many instances you want
    param: N_best: Number of instances
    type: N_best: int
    """
    df = pd.read_csv('../SVMdata/Area_R2.csv', index_col=0).head(N_best)
    svm_dictionary = {'svm': []}
    best_names = [df.iloc[i].name for i in range(N_best)]
    for s in best_names:
        svm_dictionary['svm'].append(make_instance(s, maxiter=maxiter))
    return svm_dictionary


def undersample(xTrain, yTrain):
    resample = rs(xTrain, yTrain)
    xTrain, yTrain = resample.Under()
    return xTrain, yTrain

def oversample(xTrain, yTrain):
    resample = rs(xTrain, yTrain)
    xTrain, yTrain = resample.Over()
    return  xTrain, yTrain

def smote(xTrain, yTrain):
    sm = SMOTE()
    xTrain, yTrain = sm.fit_resample(xTrain, yTrain)
    return  xTrain, yTrain

def parse_args() -> dict:
    """Parsing arguments of the command line"""
    parser = argparse.ArgumentParser('Set some values')
    parser.add_argument('--maxiter', type=int,
                        help='Choose max iterations', default=50000)
    parser.add_argument('--resampling', type=str, default='None',
                        help='Choose resampling method: undersample, oversample, smote')
    parser.add_argument('--Nbest', type=int, default='5',
                        help='Number of best models to rerun')
    args =  vars(parser.parse_args())
    return args
if __name__ == '__main__':
    
    print('Importing files')
    X, Y = get_data(normalized=False, standardized=True)
    Y = Y.flatten()
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.5)

    args = parse_args()
    maxiter = args['maxiter']
    Nbest = args['Nbest']
    try:
        xTrain, yTrain = locals()[args['resampling']](xTrain, yTrain)
    except KeyError:
        pass
    print('Import finished')

    svmdict = make_svm_dict(Nbest, maxiter=maxiter)

    Area_R2 =  {'Area_Test': {}, 'Area_Train': {},
                'R2_Test': {}, 'R2_Train':{},
                'Accuracy':{}}
    for clf in svmdict['svm']:
        start = time.time()
        g = clf.gamma
        c = clf.C
        k = clf.kernel
        d = clf.degree
        m = clf.max_iter
        filename='G:%s_C:%s_K:%s_D:%s_M:%s' %(g, c, k, d, m)

        print('Starting ', filename)
        Train, Test = predict(clf, xTrain, xTest, yTrain, yTest)
        Area_R2['Area_Test'][filename] = Test.ratio
        Area_R2['Area_Train'][filename] = Train.ratio
        Area_R2['R2_Test'][filename] = Test.R2
        Area_R2['R2_Train'][filename] = Train.R2
        Area_R2['Accuracy'][filename] = Test.acc
        Test.save_metrics('../SVMdata/', filename+str(args['resampling']))
        Train.save_metrics('../SVMdata/', filename+str(args['resampling']))
        print('accuracy: ', Test.acc)
        print('Area_Test ', Test.ratio)
        print('R2_Test ', Test.R2)
        print(filename, ' finished in {} seconds'.format(time.time() - start))
    df = pd.DataFrame(data=Area_R2)
    df.sort_values(by=['Area_Test'], ascending=False, inplace=True)
    df.to_csv('../SVMdata/Area_R2_best_data'+str(args['resampling'])+'.csv')
