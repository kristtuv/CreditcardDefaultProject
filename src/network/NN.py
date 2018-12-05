import numpy as np
import matplotlib.pylab as plt
import sys
import pandas as pd
class NeuralNet():

    def __init__(
        self,
        xData,
        yData,
        nodes=[1, 10, 1],
        activations=['sigmoid', None],
        cost_func='mse',
        regularization=None,
        lamb = 0.0):


        """
        param: xData: Data for training and testing
        param: yData: Reference data for evaluating the model
        param: nodes: Nodes in each layer. First element should
        be the size of one training example and last element depends
        on what kind of output we want. I.e for regression we have 1
        node and for classification we can have several.
        param: activations: activations in the hidden layers of your model.
        There is no activation in the input layer, and the size of activaion
        should be the size of nodes-1. The activation in the output layer
        is normally set to None.
        param: cost_func: Type of cost function to use
        param: regularization: type of regularization
        param: lamb: strength of regularization
        type: xData: ndarray
        type: yData: ndarray
        type: nodes: list
        type: activation: list
        type: cost_func: string
        type: regularization: string
        type: lamb: float
        """
        self.xData = xData
        self.yData = yData
        self.N = xData.shape[0]
        self.cost_func = cost_func
        self.regularization = regularization
        self.lamb = lamb

        if len(nodes) != (len(activations) +1):
            print("Error: invaled lengths of 'nodes' and 'activations.")
            print("Usage: nodes = [input, hidden layers (any no.), output], activations \
                    = activations between layers, len(nodes) = len(activations) + 1. \nExiting...")
            sys.exit(0)

        self.nodes = nodes
        self.activations = activations
        self.nLayers = len(activations)

        self.split_data(folds = 10, frac = 0.3, shuffle=True)
        self.initialize_weights_biases()

    def split_data(self, folds = None, frac = None, shuffle = False, resample=False):
        """
        Splits the data into training and test. Give either frac or folds

        param: folds: Number of folds
        param: frac: Fraction of data to be test data
        param: shuffle: If True: shuffles the design matrix
        type: folds: int
        type: frac: float
        type: shuffle: Bool
        return: None
        """

        if folds == None and frac == None:
            print("Error: No split info received, give either no. folds or fraction.")
            sys.exit(0)

        xData = self.xData
        yData = self.yData
        if shuffle:
            randomize = np.arange(self.N)
            np.random.shuffle(randomize)
            xData = xData[randomize]
            yData = yData[randomize]

        if folds != None:
            xFolds = np.array_split(xData, folds, axis = 0)
            yFolds = np.array_split(yData, folds, axis = 0)

            self.xFolds = xFolds
            self.yFolds = yFolds

        if frac != None:
            nTest = int(np.floor(frac*self.N))
            xTrain = xData[:-nTest]
            xTest = xData[-nTest:]

            yTrain = yData[:-nTest]
            yTest = yData[-nTest:]

            self.xTrain = xTrain ; self.xTest = xTest
            self.yTrain = yTrain ; self.yTest = yTest
            self.nTrain = xTrain.shape[0] ; self.nTest = xTest.shape[0]
            print(self.nTrain)
        if resample:
            # try:
            self.xTrain, self.yTrain = self.resample(self.xTrain, self.yTrain, oversample=True)
            self.nTrain = self.xTrain.shape[0]
            print(self.nTrain)
            # except NameError:
            #     print("Resampling requires the data to be split into training and test. Implement the frac argument")
                
    def resample(self, xTrain, yTrain, one_zero_ratio=1.0, undersample=False, oversample=True) :
        """
        Resampling the training data for binary classification with
        labels 0 and 1. Function assumes the data is skewed
        towards the 0-class and resamples from the 1-class.
        The one_zero_ratio is the ratio between classses 0 and 1
        in the new trainin set after resampling. A ratio of 1
        will yield equal amounts of each class

        param: xTrain: Training data you wish to resample
        type: xTrain: ndarray
        param: yTrain: Traininglabels you wish to resample
        type: yTrain: ndarray
        param: one_zero_ratio: ratio between class 0 and class 1 after resample
        """
        if undersample:
            #Indices of rows with class zero
            zeros_idx = np.flatnonzero(yTrain==0)
            N_one_rows = len(yTrain) - len(zeros_idx)        

            N_zero_rows = int(len(zeros_idx) - N_one_rows/one_zero_ratio)
            #Resample using indices of ones_idx
            resample_idx = np.random.choice(zeros_idx, N_zero_rows)
            xTrain = np.delete(xTrain, resample_idx, axis=0)
            yTrain = np.delete(yTrain, resample_idx, axis=0)

            return xTrain, yTrain

        if oversample:
            #Indices of rows with class one
            ones_idx = np.flatnonzero(yTrain)
            #Number of rows with class zero
            N_zero_rows = len(yTrain) - len(ones_idx)        
            #Number of new rows with class one need to give correct ratio
            N_one_rows = int(one_zero_ratio*N_zero_rows - len(ones_idx))
            #Resample using indices of ones_idx
            resample_idx = np.random.choice(ones_idx, N_one_rows)

            #New Samples 
            ySample = yTrain[resample_idx]
            xSample = xTrain[resample_idx]
            #Add samples to training set
            xTrain = np.concatenate((xTrain, xSample), axis=0)
            yTrain = np.concatenate((yTrain, ySample), axis=0)
            #Suffle training set
            shuffle_idx = np.random.permutation(np.arange(len(yTrain)) )
            xTrain = xTrain[shuffle_idx]
            yTrain = yTrain[shuffle_idx]
            return xTrain, yTrain

        





    def initialize_weights_biases(self):
        """
        Initializes weights and biases for all layers
        return: None
        """

        self.Weights = {} ; self.Biases = {}
        self.Weights_grad = {} ; self.Biases_grad = {}
        self.Z = {} ; self.A = {} ; self.C = {}

        for i in range(len(self.activations)):

            self.Weights['W'+str(i+1)] = np.random.uniform(-0.1, 0.1, (self.nodes[i], self.nodes[i+1]))
            self.Biases['B'+str(i+1)] = np.random.uniform(-0.1, 0.1, self.nodes[i+1])

            self.Weights_grad['dW'+str(i+1)] = np.zeros_like(self.Weights['W'+str(i+1)])
            self.Biases_grad['dB'+str(i+1)] = np.zeros_like(self.Biases['B'+str(i+1)])


    def activation(self, x, act_func):
        """
        Calculation of the selected
        activation function

        param: x: data
        type: x: ndarray
        param: act_func: activaion function given in init
        type: act_func: string
        return: selected activation
        """


        if act_func == 'sigmoid':

            return 1.0/(1.0 + np.exp(-x))

        elif act_func == 'tanh':

            return np.tanh(x)

        elif act_func == 'relu':

            return x * (x >= 0)

        elif act_func == None:
            return x

        else:
            print("Invalid activation function. Either 'sigmoid', 'tanh', 'relu', or None.\nExiting...")
            sys.exit(0)

    def activation_derivative(self, x, act_func):
        """
        Calculation of derivative of the selected
        activation function

        param: x: data
        type: x: ndarray
        param: act_func: activaion function given in init
        type: act_func: string
        return: derivative of activation function
        """

        if act_func == 'sigmoid':

            return x*(1 - x)

        elif act_func == 'tanh':

            return 1 - x**2

        elif act_func == 'relu':

            return 1*(x >= 0)

        elif act_func == None:
            return 1
        else:
            print("Invalid activation function. Either 'sigmoid', 'tanh', 'relu', or None.\nExiting...")
            sys.exit(0)

    def softmax(self, act):
        """
        calculates the softmax function

        param: act:
        type: act:
        return: softmax function
        """
        # Subtraction of max value for numerical stability
        act_exp = np.exp(act - np.max(act))
        self.act_exp = act_exp

        return act_exp/np.sum(act_exp, axis=1, keepdims=True)

    def cost_function(self, y, ypred):
        """
        Using the cost function defined
        in init

        param: y: correct labels
        type: y: ndarray
        param: y_pred: predicted labels
        type: y_pred: ndarray
        return: selected cost function
        """

        if self.cost_func == 'mse':
            cost =  0.5/y.shape[0]*np.sum((y - ypred)**2)

        if self.cost_func == 'log':
            cost = -0.5/y.shape[0]*np.sum(np.log(ypred[np.arange(ypred.shape[0]), y.flatten()]))

        if self.regularization == 'l2':
            for key in list(self.Weights.keys()):
                cost += self.lamb/2*np.sum(self.Weights[key]**2)

        elif self.regularization == 'l1':
            for key in list(self.Weights.keys()):
                cost += self.lamb/2*np.sum(np.abs(self.Weights[key]))

        return cost


    def cost_function_derivative(self, y, ypred):
        """
        Takes the derivative of the selected cost function

        param: y: correct labels
        type: y: ndarray
        param: y_pred: predicted labels
        type: y_pred: ndarray
        return: costfunction derivative
        """

        if self.cost_func == 'mse':
            return -1.0/y.shape[0]*(y - ypred)

        elif self.cost_func == 'log':
            ypred[np.arange(ypred.shape[0]), y.flatten()] -= 1
            return 1.0/y.shape[0]*ypred

    def accuracy(self, y, ypred):
        """
        Measures the number of correctly
        classified classes
        param: y: correct labels
        type: y: ndarray
        param: y_pred: predicted labels
        type: y_pred: ndarray
        return: accuracy
        """
        cls_pred = np.argmax(ypred, axis=1)
        return 100.0/y.shape[0]*np.sum(cls_pred == y)

    def gain_chart(self, y, ypred, threshold = 0.5):
        """
        Creates a gains chart for the predicted y-values
        when classifying a binary variable. Predicted y-values
        are given as a percent confidence of which class it
        belongs to. The percent over the given threshold is
        predicted as class 1 and everything under is given
        as class 2.

        param: y: correct label
        type: y: ndarray
        param: ypred: Predicted labels given in percent confidence
        type: ypred: ndarray
        param: threshold: number [0,1] desciding the threshold
        between the classes
        type: threshold: float
        return: None
        """

        num1 = np.sum(y)
        frac1 = num1/y.shape[0]
        sort = np.argsort(ypred[:, 1])
        lab_sort = y[sort[::-1]]

        fracs = np.arange(0, 1.01, 0.01)
        gains = np.zeros(len(fracs))

        for i in range(len(fracs)):
            lab_frac = lab_sort[:int(fracs[i]*y.shape[0])]
            gains[i] = np.sum(lab_frac)/num1

        def best(x):
            if x < frac1:
                return x/frac1
            else:
                return 1
        besty = np.zeros_like(fracs)
        for i in range(len(besty)):
            besty[i] = best(fracs[i])

        area_best = np.trapz(besty - fracs, fracs)
        area_model = np.trapz(gains - fracs, fracs)
        ratio = area_model/area_best
        print("Area ratio: ", ratio)

        plt.plot(fracs, gains, label='Lift Curve')
        plt.plot(fracs, fracs, '--', label='Baseline')
        plt.plot(fracs, besty, '--', label='Best Curve')
        plt.xlabel('Fraction of total data', fontsize=14)
        plt.ylabel('Cumulative number of target data', fontsize=14)
        plt.grid(True)
        plt.legend()
        plt.show()

    def prob_acc(self, y, ypred, n = 50):

        sort = np.argsort(ypred[:, 1])
        pred_sort = ypred[sort[::-1]]
        lab_sort = y[sort[::-1]]

        P = np.zeros((y.shape[0] - 2*n))

        for i in range(len(P)):
            P[i] = np.sum(lab_sort[n + i: 3*n + i +1])/(2*n +1)

        pred_plt = pred_sort[n:-n, 1]

        ## DO LINREG
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(pred_plt.reshape(-1, 1),P)
        b = reg.intercept_
        a = reg.coef_[0]
        R2 = reg.score(pred_plt.reshape(-1,1), P)
        print("R2 score: ", R2)

        plt.plot(pred_plt, P, 'o', markersize=0.8)
        plt.plot(pred_plt, b + pred_plt*a, label="y = %.3fx + %.3f" %(a, b))
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("Predicted probability")
        plt.ylabel("Actual probability")
        plt.grid(True)
        plt.legend()
        plt.show()


    def feed_forward(self, x, isTraining = True):
        """
        Doing the forward propagation

        param: x: Data
        type: x: ndarray
        param: isTraining: Set to false if using a finished
        model for predicting on new data.
        type: isTraining: bool
        return: activation values
        """

        self.A['A0'] = x
        for i in range(self.nLayers):

            z = self.A['A'+str(i)] @ self.Weights['W'+str(i+1)] + self.Biases['B'+str(i+1)]
            a = self.activation(z, self.activations[i])
            self.Z['Z'+str(i+1)] = z
            self.A['A'+str(i+1)] = a

        if self.cost_func == 'log':
            a = self.softmax(a)

        #self.output = a

        if isTraining:
            self.output = a
        else:
            return a



    def backpropagation(self, yTrue = None):
        """
        Function for doing the backpropagation

        param: yTrue: True values of y
        type: yTrue: ndarray
        return: None
        """
        if yTrue is None:
            yTrue = self.yTrain

        for i in range(self.nLayers, 0, -1):

            if i == self.nLayers:
                c = self.cost_function_derivative(yTrue, self.output)
            else:
                c = c @ self.Weights['W'+str(i+1)].T
                c = c * self.activation_derivative(self.A['A'+str(i)], self.activations[i-1])

            grad_w = self.A['A'+str(i-1)].T @ c
            grad_b = np.sum(c, axis= 0)

            self.Weights_grad['dW'+str(i)] = grad_w
            self.Biases_grad['dB'+str(i)] = grad_b

            if self.regularization == 'l2':
                self.Weights['W'+str(i)] -= self.eta*(grad_w + self.lamb*self.Weights['W'+str(i)])

            elif self.regularization == 'l1':
                self.Weights['W'+str(i)] -= self.eta*(grad_w + self.lamb*np.sign(self.Weights['W'+str(i)]))

            else:
                self.Weights['W'+str(i)] -= self.eta*grad_w

            self.Biases['B'+str(i)] -= self.eta*grad_b



    def TrainNN(self, epochs = 1000, batchSize = 200, eta0 = 0.01, n_print = 100):
        """
        Training the network using forward and backward propagation.

        param: epochs: Number of iterations through the entire data set
        type: epochs: int
        param: batchSize: Batch size. Must be between one and the size of the
        full data set
        type: batchSize: int
        param: eta0: learning rate or 'schedule'
        type: eta0: float, string
        param: n_print: how often we print accuracy and error to the terminal
        type: n_print: int
        return: None
        """

        if eta0 == 'schedule':
            t0 = 5 ; t1 = 50
            eta = lambda t : t0/(t + t1)
        else:
            eta = lambda t : eta0

        num_batch = int(self.nTrain/batchSize)

        self.convergence_rate = {'Epoch': [], 'Test Accuracy': []}
        for epoch in range(epochs +1):

            indices = np.random.choice(self.nTrain, self.nTrain, replace=False)

            for b in range(num_batch):

                self.eta = eta(epoch*num_batch+b)

                batch = indices[b*batchSize:(b+1)*batchSize]
                xBatch = self.xTrain[batch]
                yBatch = self.yTrain[batch]

                self.feed_forward(xBatch)
                self.backpropagation(yBatch)

            if epoch == 0 or epoch % n_print == 0:

                ypred_train = self.feed_forward(self.xTrain, isTraining=False)
                ypred_test = self.feed_forward(self.xTest, isTraining=False)
                trainError = self.cost_function(self.yTrain, ypred_train)
                testError = self.cost_function(self.yTest, ypred_test)
                print("Error after %i epochs, Training:  %g, Test:  %g" %(epoch, trainError,testError))

                if self.cost_func == 'log':
                    trainAcc = self.accuracy(self.yTrain, ypred_train)
                    testAcc = self.accuracy(self.yTest, ypred_test)
                    print("Accuracy after %i epochs, Training:   %g %%, Test:   %g %%\n" %(epoch, trainAcc, testAcc))
                    self.convergence_rate['Epoch'].append(epoch)
                    self.convergence_rate['Test Accuracy'].append(testAcc)
                    #print("-"*75)

        self.gain_chart(self.yTest, ypred_test)
        self.prob_acc(self.yTest, ypred_test)
