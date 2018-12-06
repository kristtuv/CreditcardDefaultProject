import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def get_data(normalized: bool=True,
            standardized: bool=False,
            file: str='default of credit card clients.xls'
            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetches the data for credit card defaults. Can also normalize or standardize
    the data before returning if argumetens are given
    """
    path='../credit_data/'
    fullpath=path+file
    try:
        df = pd.read_excel(fullpath)
    except: 
        df = pd.read_csv(fullpath)
    X = np.array(df.iloc[1:,:-1], dtype = np.float64)
    Y = np.array(df.iloc[1:, -1], dtype = np.int).reshape(-1, 1)

    if normalized:
        scaler = MinMaxScaler(feature_range=(0,1))
        scaledX = scaler.fit_transform(X)
        return scaledX, Y

    elif standardized:
        scaler = StandardScaler()
        scaledX = scaler.fit_transform(X)
        return scaledX, Y
    
    else:
        return X, Y

def drop_cols(save=False):
    df = pd.read_excel('../credit_data/default of credit card clients.xls')
    df = df.drop(columns=['X6', 'X7', 'X8', 'X9', 'X10', 'X11'])
    if save:
        df.to_csv('../credit_data/droppedX6-X11.csv')

if __name__=='__main__':
    print(get_data(normalized=False, standardized=False, file='droppedX6-X11.csv')[0][:10].shape)
    

