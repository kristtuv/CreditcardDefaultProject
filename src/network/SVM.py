import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pylab as plt
import time
import warnings
import argparse
import pandas as pd
import inspect

from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split, KFold

from network.newmetrics import Metrics
from read_data import get_data
warnings.filterwarnings("ignore")
sys.path.append("network/")

import os




def _convert_string_float(x):
    """ 
    Function for handling gamma input from argparser with multiple types
    used with built in map function:
    map(convert_string_float, list)
    """
    try: return float(x)
    except:
        return x 


def parse_args() -> dict:
    """Parsing arguments of the command line"""
    parser = argparse.ArgumentParser('Setting some shit')
    parser.add_argument('--kernels', nargs='*',  type=str, 
                        help='Choose one or more kernels: poly, rbf, linear', default=[None])
    parser.add_argument('--degrees', nargs='*', type=int, 
                        help='Choose degrees for poly', default=[None])
    parser.add_argument('--gammas', nargs='*', 
                        help='Choose gammas', default=[None])
    parser.add_argument('--C', nargs='*', type=float, 
                        help='Choose Cs', default=[None])
    parser.add_argument('--maxiter', nargs=1, type=int, 
                        help='Choose max iterations of svm', default=[-1])
    args =  vars(parser.parse_args())
    args['gammas'] = list(map(_convert_string_float, args['gammas']))
    return args

def svm_dict(d: dict) -> dict:
    """
    Creating a dictionary of svm instances
    """
    svm_dictionary = {'svm': []}
    max_iter=d['maxiter'][0]
    for k in d['kernels']:
        for g in d['gammas']:
            for c in d['C']:
                if k == 'rbf' or k == 'linear':
                    Svm = svm.SVC(kernel=k, C=c, gamma=g, probability=True, max_iter=max_iter)
                    svm_dictionary['svm'].append(Svm)
                else:
                    for deg in d['degrees']:
                        Svm = svm.SVC(kernel=k, C=c, gamma=g, degree=deg, probability=True, max_iter=max_iter)
                        svm_dictionary['svm'].append(Svm)
    return svm_dictionary

def svm_parameters():
    args = parse_args()
    defaults = {'C': [10, 1.0, 0.1, 0.01, 0.001, 0.0001],
                'gammas':['scale', 1, 0.1, 0.01, 0.001, 0.0001],
                'kernels': ['rbf', 'poly', 'linear'],
                'degrees': np.arange(11)
        }
     
    for i, j in args.items():
        if j[0] != None:
            defaults[i] = j
    return defaults

def predict(clf, xTrain, xTest, yTrain, yTest):
    # print('GAMMA: ', clf.gamma, 'C: ', clf.C, 'KERNEL: ', clf.kernel, 'DEGREE: ', clf.degree)
    clf.fit(xTrain, yTrain)
    ypred_train = clf.predict_proba(xTrain)
    ypred_test = clf.predict_proba(xTest)

    Train = Metrics(yTrain, ypred_train)
    Train.gain_chart(plot=False)
    Train.prob_acc(plot=False)

    Test = Metrics(yTest, ypred_test)
    Test.gain_chart(plot=False)
    Test.prob_acc(plot=False)
    return Train, Test

if __name__ == '__main__':
    pass
