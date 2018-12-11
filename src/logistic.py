import numpy as np
from read_data import get_data
import sys
sys.path.append("network/")
from logreg import LogReg
from metrics import gain_chart, prob_acc

#X, Y = get_data(normalized = False, standardized = False)
X, Y = get_data()

logreg = LogReg(X, Y)
logreg.optimize(m = 100, epochs= 2000, eta = 0.01, regularization='l2', lamb=0.0001)

ypred_train = logreg.p_train
ypred_test = logreg.p_test
gain_chart(logreg.Y_test, ypred_test)
prob_acc(logreg.Y_test, ypred_test)
