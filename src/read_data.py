import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def get_data(normalized: bool=True,
            standardized: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fetches the data for credit card defaults. Can also normalize or standardize
    the data before returning if argumetens are given
    """
    df = pd.read_excel('../credit_data/default of credit card clients.xls')
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



if __name__=='__main__':
    pass
    # get_data(normalized=False, standardized=False, resample=True)
