import numpy as np
class Resample:
    def __init__(self, xTrain, yTrain):
        self.xTrain = xTrain
        self.yTrain = yTrain

    def Under(self, ratio=1.0):
        xTrain = self.xTrain
        yTrain = self.yTrain

        #Indices of rows with class zero
        zeros_idx = np.flatnonzero(yTrain==0)

        #Number of rows
        N_zero_rows = len(zeros_idx)
        N_one_rows = len(yTrain) - len(zeros_idx)        
        N_sub_zero_rows = int(len(zeros_idx) - N_one_rows/ratio)

        #Resample using indices of ones_idx
        resample_idx = np.random.choice(zeros_idx, N_zero_rows)
        xTrain = np.delete(xTrain, resample_idx, axis=0)
        yTrain = np.delete(yTrain, resample_idx, axis=0)

        #Shuffle training set
        shuffle_idx = np.random.permutation(np.arange(len(yTrain)) )
        xTrain = xTrain[shuffle_idx]
        yTrain = yTrain[shuffle_idx]

        return xTrain, yTrain
    def Over(self, ratio=1.0):
        xTrain = self.xTrain
        yTrain = self.yTrain

        #Indices of rows with class one
        ones_idx = np.flatnonzero(yTrain)

        #Number of rows
        N_one_rows = len(ones_idx)
        N_zero_rows = len(yTrain) - N_one_rows
        N_add_one_rows = int(ratio*N_zero_rows - len(ones_idx))

        #Resample using indices of ones_idx
        resample_idx = np.random.choice(ones_idx, N_add_one_rows)

        #New Samples 
        ySample = yTrain[resample_idx]
        xSample = xTrain[resample_idx]

        #Add samples to training set
        xTrain = np.concatenate((xTrain, xSample), axis=0)
        yTrain = np.concatenate((yTrain, ySample), axis=0)

        #Shuffle training set
        shuffle_idx = np.random.permutation(np.arange(len(yTrain)) )
        xTrain = xTrain[shuffle_idx]
        yTrain = yTrain[shuffle_idx]
        return xTrain, yTrain

if __name__=='__main__':
   pass 

