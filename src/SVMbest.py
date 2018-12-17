import pandas as pd
import numpy as np
import re
import time
import sys

from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split, KFold

from network.SVM import predict
from network.SVM import _convert_string_float as csf
from read_data import get_data
from network.NN import NeuralNet

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




if __name__ == '__main__':
    print('Importing files')
    X, Y = get_data(normalized=False, standardized=True)
    Y = Y.flatten()
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.5)
    print('Import finished')
    try:
        maxiter = int(sys.argv[1])
    except IndexError:
        maxiter = 1000000

    svmdict = make_svm_dict(1, maxiter=maxiter)

    Area_R2 =  {'Area_Test': {}, 'Area_Train': {}, 'R2_Test': {}, 'R2_Train':{}}
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
        print(filename+'accuracy: ', Test.acc)
        Area_R2['Area_Test'][filename] = Test.ratio
        Area_R2['Area_Train'][filename] = Train.ratio
        Area_R2['R2_Test'][filename] = Test.R2
        Area_R2['R2_Train'][filename] = Train.R2
        Test.save_metrics('../SVMdata/', filename)
        Train.save_metrics('../SVMdata/', filename)
        print(filename, ' finished in {} seconds'.format(time.time() - start))
    df = pd.DataFrame(data=Area_R2)
    df.sort_values(by=['Area_Test'], ascending=False, inplace=True)
    df.to_csv('../SVMdata/Area_R2_best_data.csv')
