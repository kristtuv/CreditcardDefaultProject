import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pylab as plt

from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split, KFold
from metrics import gain_chart, prob_acc
from read_data import get_data
import warnings
warnings.filterwarnings("ignore")
        
sys.path.append("network/")

X, Y = get_data(normalized=False, standardized=True)
Y = Y.flatten()
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.5)


# clf = svm.SVC(gamma='scale', kernel='poly', degree=15, probability=True, max_iter=10)#, class_weight={0: 20})
# clf = svm.SVC(gamma='scale', kernel='poly', C=1, coef0=1, degree=5, probability=True, max_iter=20)#, class_weight={0: 20})
# clf = svm.SVC(gamma=0.1, kernel='rbf', C=0.001, probability=True, max_iter=20)#, class_weight={0: 20})
gamma = [1, 0.1, 0.01, 0.001, 0.0001]
C = [10, 1.0, 0.1, 0.01, 0.001, 0.0001]
kernels = ['linear', 'rbf', 'poly']
degrees = np.arange(1, 11)
# gamma = [0.1]
# C = [0.1, 0.01]
dic = {'clf': []}
for g in gamma:
    for c in C:
        for k in kernels:
            for d in degrees:
                clf = svm.SVC(kernel=k, degree=d, C=c, gamma=g, probability=True)
        # clf = svm.SVC(kernel='rbf', gamma=g, C=c, probability=True)
                dic['clf'].append(clf)

for i in dic['clf']:
    i.fit(xTrain, yTrain)
    print('GAMMA: ', i.gamma, 'C: ', i.C, 'KERNEL: ', i.kernel, 'DEGREE: ', i.degree)
    ypred_train = i.predict_proba(xTrain)
    ypred_test = i.predict_proba(xTest)
    gain_chart(yTrain, ypred_train, plot=False)
    gain_chart(yTest, ypred_test, plot=False)
    prob_acc(yTrain, ypred_train, plot=False)
    prob_acc(yTest, ypred_test, plot=False)

# # clf.fit(xTrain, yTrain)

# # ypred_train = clf.predict_proba(xTrain)
# # ypred_test = clf.predict_proba(xTest)
# # gain_chart(yTrain, ypred_train, plot=False)
# # gain_chart(yTest, ypred_test, plot=True)
# # prob_acc(yTrain, ypred_train, plot=False)
# # prob_acc(yTest, ypred_test, plot=True)


