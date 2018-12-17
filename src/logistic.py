import numpy as np
from read_data import get_data
import sys
sys.path.append("network/")
from logreg import LogReg
from metrics import gain_chart, prob_acc

#X, Y = get_data(normalized = False, standardized = False)
X, Y = get_data()

logreg = LogReg(X, Y)
logreg.optimize(m = 100, epochs= 5000, eta = 0.01)#, regularization='l2', lamb=0.0001)

ypred_train = logreg.p_train
ypred_test = logreg.p_test
gain_chart(logreg.Y_test, ypred_test)
prob_acc(logreg.Y_test, ypred_test)

def regularization_test():

    with open("log_regular.txt", "w+") as f:

        f.write("Regularization | lambda | Area | R2\n")
        f.write("-"*50 + "\n")
        for reg in ['l2', 'l1']:
            for lamb in [0.0001, 0.001, 0.01, 0.1, 1.0]:

                logreg = LogReg(X, Y)
                logreg.optimize(m = 100, epochs= 5000, eta = 0.01, regularization=reg, lamb=lamb)

                ypred_train = logreg.p_train
                ypred_test = logreg.p_test
                area = gain_chart(logreg.Y_test, ypred_test, plot = False)
                R2 = prob_acc(logreg.Y_test, ypred_test, plot = False)

                f.write("  %s  |  %g  |  %.4f  |  %.4f  \n" %(reg, lamb, area, R2))

            f.write("-"*50 + "\n")

if __name__ == "__main__":
    regularization_test()
