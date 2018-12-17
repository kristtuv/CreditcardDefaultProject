import numpy as np
import sys
sys.path.append("../network/")
sys.path.append("../")

from read_data import get_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate, train_test_split, KFold
from metrics import gain_chart, prob_acc
from sklearn.tree import DecisionTreeClassifier
from resampling import Resample

X, Y = get_data()
Y = Y.flatten()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)
#r = Resample(X_train, Y_train)
#X_train, Y_train = r.Over()


clf_rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_split = 100)

clf_rf.fit(X_train, Y_train)

ypred_test = clf_rf.predict_proba(X_test)

gain_chart(Y_test, ypred_test)
prob_acc(Y_test, ypred_test)



clf_dt = DecisionTreeClassifier(max_depth = 6, min_samples_split = 200)
clf_dt.fit(X_train, Y_train)

ypred_test = clf_dt.predict_proba(X_test)

gain_chart(Y_test, ypred_test)
prob_acc(Y_test, ypred_test)
