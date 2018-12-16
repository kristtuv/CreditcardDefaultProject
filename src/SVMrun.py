from network.SVM import *

if __name__ == '__main__':
    parameters = svm_parameters()
    
    print('Importing files')
    X, Y = get_data(normalized=False, standardized=True)
    Y = Y.flatten()
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.5)
    print('Import finished')

    Area_R2 =  {'Area_Test': {}, 'Area_Train': {}, 'R2_Test': {}, 'R2_Train':{}}

    for clf in svm_dict(parameters)['svm']:
        start = time.time()
        g = clf.gamma
        c = clf.C
        k = clf.kernel
        d = clf.degree
        m = clf.max_iter
        filename='G:%s_C:%s_K:%s_D:%s_M:%s' %(g, c, k, d, m)

        print('Starting ', filename)
        Train, Test = predict(clf, xTrain, xTest, yTrain, yTest)
        Area_R2['Area_Test'][filename] = Test.ratio
        Area_R2['Area_Train'][filename] = Train.ratio
        Area_R2['R2_Test'][filename] = Test.R2
        Area_R2['R2_Train'][filename] = Train.R2
        print(filename, ' finished in {} seconds'.format(time.time() - start))
    df = pd.DataFrame(data=Area_R2)
    df.sort_values(by=['Area_Test'], ascending=False, inplace=True)
    df.to_csv('../SVMdata/Area_R2.csv')


