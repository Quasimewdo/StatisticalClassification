import numpy as np
import matplotlib.pyplot as plt
from statistics import stdev
from statistics import mean
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import DotProduct

# import some data to play with
digits = datasets.load_digits()


def loss_function(Theta, x, y):
# Returns the loss_value vector and the index j_star where j_star is the argmax of loss vector
    d, K = Theta.shape
    loss_vector = np.maximum(np.zeros([K, 1]), (np.ones([K, 1]) + x.transpose().dot(Theta - np.reshape(Theta[:,y], [-1, 1])).transpose()))
    loss_vector[y] = -1
    print(loss_vector)
    j_star = np.argmax(loss_vector)
    loss_value = loss_vector[j_star][0]
    print(loss_value)
    return loss_value, j_star

def subgradient(Theta, x, y):
# Computes a subgrafient of the objective empirical hinge loss
# one pair of (x, y) given, x of size 1, and y an integer in {0, 1, ..., 9}.
    g = np.zeros(Theta.shape)
    x = np.reshape(x, [-1, 1])
    y = int(y)
    d, K = Theta.shape
    loss, j_star = loss_function(Theta, x, y)
    if loss > 0:
        g[:, j_star] = x.transpose()
        g[:, y] = -x.transpose()
    return(g)

def svmsubgradient(Theta, x, y):
#  Returns a subgradient of the objective empirical hinge loss
# The inputs are Theta, of size n-by-K, where K is the number of classes,
# x of size n, and y an integer in {0, 1, ..., 9}.
    G = np.zeros(Theta.shape)
    for x_k, y_k in zip(x, y):
        G += subgradient(Theta, x_k, y_k)
    G /= x.shape[0]
    return(G)

def sgd(Xtrain, ytrain, maxiter = 10, init_stepsize = 1.0, l2_radius = 10000, alpha = 0.5): 
#
# Performs maxiter iterations of projected stochastic gradient descent
# on the data contained in the matrix Xtrain, of size n-by-d, where n
# is the sample size and d is the dimension, and the label vector
# ytrain of integers in {0, 1, ..., 9}. Returns two d-by-10
# classification matrices Theta and mean_Theta, where the first is the final
# point of SGD and the second is the mean of all the iterates of SGD.
#
# Each iteration consists of choosing a random index from n and the
# associated data point in X, taking a subgradient step for the
# multiclass SVM objective, and projecting onto the Euclidean ball
# The stepsize is init_stepsize / sqrt(iteration).
    K = 10
    NN, dd = Xtrain.shape
    print(NN)
    Theta = np.zeros(dd*K)
    Theta.shape = dd,K
    mean_Theta = np.zeros(dd*K)
    mean_Theta.shape = dd,K
    ## YOUR CODE HERE -- IMPLEMENT PROJECTED STOCHASTIC GRADIENT DESCENT
    for i in range(maxiter):
        index = np.random.randint(0, NN)
        stepsize = init_stepsize / ((i+1)**alpha)
        Theta -= stepsize * subgradient(Theta, Xtrain[index,:], ytrain[index])
        radius = np.linalg.norm(Theta.flatten())
        if radius > l2_radius:
            Theta = Theta / radius * l2_radius
        mean_Theta += Theta
    mean_Theta /= maxiter
    return Theta, mean_Theta

def Classify(Xdata, Theta):
#
# Takes in an N-by-d data matrix Adata, where d is the dimension and N
# is the sample size, and a classifier X, which is of size d-by-K,
# where K is the number of classes.
#
# Returns a vector of length N consisting of the predicted digits in
# the classes.
    scores = np.matmul(Xdata, Theta)
    inds = np.argmax(scores, axis = 1)
    return(inds)

#choose a seed
seed = 1
np.random.seed(seed)
# Load data into train set and test set
digits = datasets.load_digits()
X = digits.data
y = np.array(digits.target, dtype = int)
X,y = shuffle(X,y)
N,d = X.shape
Ntest = np.int(100)

Xtrain = X[0:10,:]
ytrain = y[0:10]
Xtest = X[10:N,:]
ytest = y[10:N]

alpha = 0.5
l2_radius = 40.0
M_raw = np.sqrt(np.mean(np.sum(np.square(Xtrain))))
init_stepsize = l2_radius/M_raw
maxiter = 40000
Theta, mean_Theta = sgd(Xtrain, ytrain, maxiter, init_stepsize, l2_radius, alpha)


"""
###### Question d : performance vs size of training sets
error_vector = np.zeros([6, 5])
Ntrains = [20, 50, 100, 500, 1000, 1500]
for i in range(len(Ntrains)):
    Xtrain = X[0:Ntrains[i],:]
    ytrain = y[0:Ntrains[i]]
    Xtest = X[Ntrains[i]:N,:]
    ytest = y[Ntrains[i]:N]

    alpha = 0.5
    l2_radius = 40.0
    M_raw = np.sqrt(np.mean(np.sum(np.square(Xtrain))))
    init_stepsize = l2_radius/M_raw
    maxiter = 40000
    for j in range(5):
        Theta, mean_Theta = sgd(Xtrain, ytrain, maxiter, init_stepsize, l2_radius, alpha)
        error_vector[i, j] = np.sum(np.not_equal(Classify(Xtest, mean_Theta),ytest)/Ntest)

std = np.zeros([6, 1])
avg = np.zeros([6, 1])
for i in range(len(Ntrains)):
    std[i] = stdev(error_vector[i])
    avg[i] = mean(error_vector[i])

fig, ax = plt.subplots()
ax.errorbar(Ntrains, avg,
            yerr=std,
            fmt='-o')
ax.set_xlabel('Number of train samples')
ax.set_ylabel('Error')
ax.set_title('Errors of model depending on the number of training sets')
plt.show()

###### Question e : Performance vs learning rate
Ntrain = np.int(1697)
Xtrain = X[0:Ntrain,:]
ytrain = y[0:Ntrain]
Xtest = X[Ntrain:N,:]
ytest = y[Ntrain:N]

error_alpha_vector = np.zeros([10, 5])

alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for i in range(len(alphas)):
    alpha = alphas[i]
    l2_radius = 40.0
    M_raw = np.sqrt(np.mean(np.sum(np.square(Xtrain))))
    init_stepsize = l2_radius/M_raw
    maxiter = 40000
    for j in range(5):
        Theta, mean_Theta = sgd(Xtrain, ytrain, maxiter, init_stepsize, l2_radius, alpha)
        error_alpha_vector[i, j] = np.sum(np.not_equal(Classify(Xtest, mean_Theta),ytest)/Ntest)

std = np.zeros([10, 1])
avg = np.zeros([10, 1])
for i in range(len(alphas)):
    std[i] = stdev(error_alpha_vector[i])
    avg[i] = mean(error_alpha_vector[i])

fig, ax = plt.subplots()
ax.errorbar(alphas, avg,
            yerr=std,
            fmt='-o')
ax.set_xlabel('alpha')
ax.set_ylabel('Error')
ax.set_title('Errors of model depending on the stepsize parameter alpha')
plt.show()
"""