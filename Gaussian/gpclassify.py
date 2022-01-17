"""
=====================================================
Gaussian process classification (GPC)
=====================================================

"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel, Matern, WhiteKernel, RationalQuadratic, ExpSineSquared
from sklearn.preprocessing import LabelBinarizer

def return_onehot(y):
    #Note, use LabelBinarizer instead of OneHotEncoder
    lb = LabelBinarizer()
    lb.fit(range(0,10,1))
    yonehot=lb.transform(y)
    #print(np.size(yonehot, 0),np.size(yonehot, 1) )
    return yonehot

def train_model_GPClass(Xtrain, ytrain, kernel, Xtest, ytest):
    #gpc_rbf = GaussianProcessClassifier(kernel=kernel).fit(Xtrain, ytrain)
    gpc_rbf = GaussianProcessClassifier(kernel=kernel, optimizer = None).fit(Xtrain, ytrain)
    #GaussianProcessClassifier performs hyperparameter estimation, this means that the the value specified above may not be the final hyperparameters
    #If you dont want it to do hyperparameter optimization set optimizer = None like this GaussianProcessClassifier(kernel=kernel,optimizer = None).fit(Xtrain, ytrain)


    yp_train = gpc_rbf.predict(Xtrain)
    train_error_rate = np.mean(np.not_equal(yp_train,ytrain))

    yp_test = gpc_rbf.predict(Xtest)
    test_error_rate = np.mean(np.not_equal(yp_test,ytest))

    kernel_params = gpc_rbf.kernel_.get_params()['kernels']

    return train_error_rate, test_error_rate, str(kernel_params[0])

def train_model_GPRegr(Xtrain, ytrain, kernel, Xtest, ytest):
    #transform to onehot
    yonehot = return_onehot(ytrain)

    gpr = GaussianProcessRegressor(kernel=kernel).fit(Xtrain, yonehot)

    yp_train = gpr.predict(Xtrain)
    yp_train_indx = np.argmax(yp_train, axis = 1) #Find max of each row, i.e find class

    train_error_rate = np.mean(np.not_equal(yp_train_indx,ytrain))

    yp_test = gpr.predict(Xtest)
    yp_test_indx = np.argmax(yp_test, axis = 1) #Find class
    test_error_rate = np.mean(np.not_equal(yp_test_indx,ytest))

    #print(gpr.get_params())
    kernel_param = gpr.get_params()['kernel']

    return train_error_rate, test_error_rate, str(kernel_param)

def calculate_new_values(error_df, methods, training_sizes, kernels, iters = 3):

    # import data
    digits = datasets.load_digits()

    X = digits.data
    y = np.array(digits.target, dtype = int)
    N,d = X.shape

    for method in methods:
        for kernel_ind in range(len(kernels)):
            kernel = kernels[kernel_ind]

            for size_ind in range(len(training_sizes)):
                Ntrain = training_sizes[size_ind]
                temporary_df = error_df[(error_df['method'] == method) & (error_df['Ntrain'] == Ntrain) &(error_df['kernel'] == str(kernel) )]
                #Check that the value is not already calculated
                if temporary_df.empty:

                    print('method = %s, Ntrain = %d, kernel = %s has not been prev. calc.' %(method, Ntrain,str(kernel)) )
                    # Add loop to use avg error insted. add shuffle here
                    try:
                        mean_test_error = 0
                        mean_train_error = 0
                        for i in range(iters): #do the errorcalculation multiple times, take average
                            X,y = shuffle(X,y)
                            Xtrain = X[0:Ntrain-1,:]
                            ytrain = y[0:Ntrain-1]
                            Xtest = X[Ntrain:N,:]
                            ytest = y[Ntrain:N]

                            if method == 'GPC':
                                train_error_rate, test_error_rate, kernel_params = train_model_GPClass(Xtrain, ytrain, kernel, Xtest, ytest) #a)
                            elif method == 'GPR':
                                train_error_rate, test_error_rate, kernel_params = train_model_GPRegr(Xtrain, ytrain, kernel, Xtest, ytest) #b)
                            else:
                                raise ValueError("%s is not an acceptable method" %method)

                            mean_train_error += train_error_rate / iters
                            mean_test_error += test_error_rate/iters
                        #     print(test_error_rate)
                        # print(mean_test_error)
                        print("Success when computing model")
                        row_df = {'method': method, 'Ntrain': Ntrain, 'kernel':kernel_params, 'train_error': mean_train_error , 'test_error': mean_test_error}
                        error_df = error_df.append(row_df, ignore_index=True)

                    except:
                        print("Error when computing model")
                else:
                    pass

    return error_df

def plot(error_df, methods, training_sizes, kernels):
    #Get the subdataframe to be plotted
    kernel_names = [str(kernels[i]) for i in range(len(kernels))]
    sub_df = error_df[(error_df['method'].isin(methods)) & (error_df['Ntrain'].isin(training_sizes)) &(error_df['kernel'].isin (kernel_names))]

    #plot it
    fig = px.line(sub_df, x = 'Ntrain', y = 'test_error', color = 'kernel', line_dash = 'method', log_x = True, log_y = True)

    img_name = "images/" + "".join(methods) + str(kernels[0]) + ".png"
    fig.write_image(img_name)
    fig.show()
    return

def main():
    #choose a seed
    seed = 1
    np.random.seed(seed)

    training_sizes = [20, 50, 100, 500, 1000, 1500]
    #training_sizes = [20, 50, 100, 500]

    ##SPECIFY WHICH KERNELS TO USE
    kernels = []
    #kernels = [DotProduct(1.0), 1.0 * RBF([10]),  DotProduct() + WhiteKernel(10), Matern(7.5, nu = 0.5)];
    #kernels = [DotProduct() + ConstantKernel() +WhiteKernel(10), Matern(), RationalQuadratic(), ExpSineSquared()]

    params = [1,7.5,10,25,75]
    nu_vals = [ 0.5, 1.5, 2.5, 10]
    dot_params = [0,1e-5,1,10,100]
    for param in params:
        kernels.append(1.0*RBF(param))

    # for nu_val in nu_vals:
    #     kernels.append(Matern(7.5, nu = nu_val))
    # kernels.append(RBF(7.5))
    #
    # for param in dot_params:
    # #    kernels.append(DotProduct(param))
    #      kernels.append(DotProduct(10) + WhiteKernel(param))

    ##SPECIFY WHICH METHODS TO USE
    #methods = ['GPR']
    methods = ['GPR', 'GPC']

    ##USE DATAFRAME TO SAVE CALCULATIONS
    #error_df = pd.DataFrame(columns=['method', 'Ntrain', 'kernel', 'train_error', 'test_error']) #Use if wanted to start from beginning
    error_df = pd.read_csv('error.csv', index_col = 0)

    error_df = calculate_new_values(error_df, methods, training_sizes, kernels)
    error_df.to_csv("error.csv")
    # print(error_df)

    plot(error_df, methods, training_sizes, kernels)

    return

main()
