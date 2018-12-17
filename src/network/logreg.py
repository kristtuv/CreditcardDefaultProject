import numpy as np
from sklearn import linear_model
import matplotlib.pylab as plt
import pickle,os
from sklearn.model_selection import train_test_split


class LogReg():

    def __init__(self, X, Y, test_size = 0.5):

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)

        self.X_train = X_train ; self.X_test = X_test
        self.Y_train = Y_train ; self.Y_test = Y_test

        # Initialize parameters
        self.beta = np.random.uniform(-0.5, 0.5, (self.X_train.shape[1],1))

    def optimize(
        self,
        method = 'SGD',
        m = 100,
        epochs = 100,
        eta = 'schedule',
        regularization=None,
        lamb = 0.0):

        """
        param: method: What type of gradient descent optimiser
        param: m: Iterations within one epoch
        param: epochs: Number of full iterations through data set
        param: eta: Learing rate
        param: regularization: Type of regularization
        param: lamb: Regularization strength
        type: method: string
        type: m: int
        type: epochs: int
        type: eta: string
        type: regularization: string
        type: lamb: float
        """

        X_train = self.X_train ; X_test = self.X_test
        Y_train = self.Y_train ; Y_test = self.Y_test
        beta = self.beta

        batchSize = int(X_train.shape[0]/m)
        self.batchSize = batchSize
        print("Batch size: ", batchSize)

        if eta == 'schedule':
            t0 = 5 ; t1 = 50
            learning_schedule = lambda t : t0/(t + t1)
        else:
            learning_schedule = lambda t : eta

        if regularization == 'l2':
            reg_cost = lambda beta: lamb*np.sum(beta**2)
            reg_grad = lambda beta: 2*lamb*beta
        elif regularization == 'l1':
            reg_cost = lambda beta: lamb*np.sum(np.abs(beta))
            reg_grad = lambda beta: lamb*np.sign(beta)
        elif regularization == None:
            reg_cost = lambda beta: 0
            reg_grad = lambda beta: 0

        #Stochastic Gradient Descent (SGD)
        for epoch in range(epochs + 1):

            # Shuffle training data
            randomize = np.arange(X_train.shape[0])
            np.random.shuffle(randomize)
            X_train = X_train[randomize]
            Y_train = Y_train[randomize]

            for i in range(m):

                rand_idx = np.random.randint(m)

                xBatch = X_train[rand_idx*batchSize : (rand_idx+1)*batchSize]
                yBatch = Y_train[rand_idx*batchSize : (rand_idx+1)*batchSize]

                y = xBatch @ beta
                #p = np.exp(y)/(1 + np.exp(y))
                p = 1.0/(1 + np.exp(-y))
                eta = learning_schedule(epoch*m+i)
                dbeta = -(xBatch.T @ (yBatch - p))/xBatch.shape[0] + reg_grad(beta)
                beta -= eta*dbeta

            if epoch % 10 == 0 or epoch == 0:

                logit_train = X_train @ beta
                self.train_cost = 0.5*(-np.sum((Y_train * logit_train) - np.log(1 + np.exp(logit_train))))/X_train.shape[0] + reg_cost(beta)
                self.p_train = 1/(1 + np.exp(-logit_train))
                self.train_accuracy = np.sum((self.p_train > 0.5) == Y_train)/X_train.shape[0]

                logit_test = X_test @ beta
                self.test_cost = 0.5*(-np.sum((Y_test * logit_test) - np.log(1 + np.exp(logit_test))))/X_test.shape[0] + reg_cost(beta)
                self.p_test = 1/(1 + np.exp(-logit_test))
                self.test_accuracy = np.sum((self.p_test > 0.5) == Y_test)/X_test.shape[0]

                print("Epoch: ", epoch)
                print("  Cost  | Training: %f, Test: %f" %(self.train_cost, self.test_cost))
                print("Accuracy| Training: %f, Test: %f" %(self.train_accuracy, self.test_accuracy))
                print("-"*50)


if __name__ == '__main__':

    logreg = LogReg()
    logreg.optimize(m = 1000, regularization='l1', lamb = 100)
