import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pylab as plt
import time
import warnings
import argparse

from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split, KFold
from metrics import gain_chart, prob_acc
from read_data import get_data
warnings.filterwarnings("ignore")
sys.path.append("network/")
import signal
import os
from functools import wraps

def timeout(timeout_seconds):
    def decorator(function):
        msg = 'Function (%s) used too much time (%s sec)' % (function.__name__, timeout_seconds)

        def handler(signum, frame):
            """If the signum event happens, a TimeoutError is raised"""
            raise TimeoutError(msg)
        def _f(*args, **kwargs):
            signal.alarm(0)
            old = signal.signal(signal.SIGALRM, handler) #Store the default handler of SIGALRM
            signal.alarm(timeout_seconds) #Seconds before SIGALRM signal is set
            try:
                #Will go to finally block when SIGALRM is set
                # after timeout_seconds nr of seconds
                #because SIGLRM raises TimeoutError
                print(os.getpid())
                function_result = function(*args, **kwargs) 
            finally:
                signal.signal(signal.SIGALRM, old) # Reset handler to default
            signal.alarm(0) #Reset signal to default
            return function_result
        _f.__name__ = function.__name__
        return _f
    return decorator


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
    args =  vars(parser.parse_args())
    args['gammas'] = list(map(_convert_string_float, args['gammas']))
    return args

def svm_dict(d: dict) -> dict:
    """
    Creating a dictionary of svm instances
    """
    svm_dictionary = {'svm': []}

    for k in d['kernels']:
        for g in d['gammas']:
            for c in d['C']:
                if k == 'rbf' or k == 'linear':
                    Svm = svm.SVC(kernel=k, C=c, gamma=g, probability=True, max_iter=5000)
                    svm_dictionary['svm'].append(Svm)
                else:
                    for d in d['degrees']:
                        Svm = svm.SVC(kernel=k, C=c, gamma=g, degree=d, probability=True)
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
    print('in predict')
    # dummy()
    # print('done with dummy')
    # print('GAMMA: ', clf.gamma, 'C: ', clf.C, 'KERNEL: ', clf.kernel, 'DEGREE: ', clf.degree)
    print('fittin')
    clf.fit(xTrain, yTrain)
    print('done fittin')
    # print('Fit done')
    # ypred_train = clf.predict_proba(xTrain)
    # ypred_test = clf.predict_proba(xTest)
    
    
    # gain_chart(yTrain, ypred_train, plot=False)
    # gain_chart(yTest, ypred_test, plot=False)
    # prob_acc(yTrain, ypred_train, plot=False)
    # prob_acc(yTest, ypred_test, plot=False) 


if __name__ == '__main__':
    print(os.getpid())
    parameters = svm_parameters()
    print(parameters)
    X, Y = get_data(normalized=False, standardized=True)
    Y = Y.flatten()
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.5)
    print(svm_dict(parameters)['svm']) 
    for i, clf in enumerate(svm_dict(parameters)['svm']):
    # for i in range(10):
        print(i)
        try:
            predict(clf, xTrain, xTest, yTrain, yTest)

        except TimeoutError as TE:
            print(str(TE))
            continue
        # try:
        #     predict(clf, xTrain, xTest, yTrain, yTest)

        # except TimeoutError:
        #     # print(str(TE))
        #     print('hei')
        #     continue




