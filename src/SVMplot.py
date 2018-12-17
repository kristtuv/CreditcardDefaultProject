import matplotlib.pyplot as plt
import sys
from os import listdir
import pandas as pd
from network.newmetrics import Metrics as M

def gains():
    for f in gain_files:
        print(f)
        df = pd.read_csv(f, index_col=0)
        fracs = df['fracs']
        besty = df['besty']
        gains = df['gains']
        M().plot_gains(fracs, gains, besty)
def prob():
    for f in prob_files:
        print(f)
        df = pd.read_csv(f, index_col=0)
        pred_plt = df['pred_plt']
        P = df['P']
        b = df['b']
        a = df['a']
        M().plot_acc(pred_plt, P, b[1], a[1])
if __name__ == '__main__':
    prob_files = ['../SVMdata/{}'.format(f) for f in listdir('../SVMdata/')
            if f.startswith('G') and f.endswith('prob.csv')]
    gain_files = ['../SVMdata/{}'.format(f) for f in listdir('../SVMdata/')
            if f.startswith('G') and f.endswith('gain.csv')]
    
    #gains()
    prob()
