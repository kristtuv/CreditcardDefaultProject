import numpy as np
from read_data import get_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, train_test_split, KFold
from metrics import gain_chart, prob_acc
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pylab as plt
from tqdm import tqdm
import sys
sys.path.append("network/")

X, Y = get_data()
Y = Y.flatten()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

depths = np.arange(2, 20, 2)
n_splits = 10
cv_gains_rf = np.zeros((2, len(depths)))
cv_probs_rf = np.zeros((2, len(depths)))
cv_gains_dt = np.zeros((2, len(depths)))
cv_probs_dt = np.zeros((2, len(depths)))

for i in tqdm(range(len(depths))):

    clf_rf = RandomForestClassifier(n_estimators=100, max_depth=depths[i])
    clf_dt = DecisionTreeClassifier(max_depth = depths[i])

    kf = KFold(n_splits=n_splits)

    for train,valid in kf.split(X_train):

        # RANDOM FOREST
        clf_rf.fit(X_train[train], Y_train[train])

        ypred_train = clf_rf.predict_proba(X_train[train])
        ypred_test = clf_rf.predict_proba(X_train[valid])

        cv_gains_rf[0,i] += gain_chart(Y_train[train], ypred_train, plot=False)
        cv_gains_rf[1,i] += gain_chart(Y_train[valid], ypred_test, plot=False)
        cv_probs_rf[0,i] += prob_acc(Y_train[train], ypred_train, plot=False)
        cv_probs_rf[1,i] += prob_acc(Y_train[valid], ypred_test, plot=False)

        # REGULAR CLASSIFICATION TREE
        clf_dt.fit(X_train[train], Y_train[train])

        ypred_train = clf_dt.predict_proba(X_train[train])
        ypred_test = clf_dt.predict_proba(X_train[valid])

        cv_gains_dt[0,i] += gain_chart(Y_train[train], ypred_train, plot=False)
        cv_gains_dt[1,i] += gain_chart(Y_train[valid], ypred_test, plot=False)
        cv_probs_dt[0,i] += prob_acc(Y_train[train], ypred_train, plot=False)
        cv_probs_dt[1,i] += prob_acc(Y_train[valid], ypred_test, plot=False)

cv_gains_rf /= n_splits ; cv_probs_rf /= n_splits
cv_gains_dt /= n_splits ; cv_probs_dt /= n_splits

for y, ylab in zip([cv_gains_rf.T, cv_gains_dt.T, cv_probs_rf.T, cv_probs_dt.T], \
                ['Area Ratio', 'Area Ratio', 'R2 Score', 'R2 Score']):
    plt.plot(depths, y)
    plt.legend(['Training', 'Validation'])
    plt.xlabel('Depth', fontsize=14)
    plt.ylabel(ylab, fontsize=14)
    plt.show()
