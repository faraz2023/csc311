from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing
from scipy.special import logsumexp
from sklearn import linear_model
from sklearn.model_selection import KFold

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

np.random.seed(42)

idx = np.random.permutation(range(N))

def general_normalizer(X, vector=False):
    if vector:
        return preprocessing.normalize(X[:,np.newaxis], axis=0).ravel()
    return preprocessing.normalize(X, axis=0)

#helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist
 
#to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Given a test datum, it returns its prediction based on locally weighted regression

    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    n = x_train.shape[0]
    x_l2_distance = l2(x_train, test_datum.transpose())

    # x_l2_distance is a vector that contains the l2 distance from the test vector to each of the train data points
    x_l2_distance =x_l2_distance[:,0]

    # calculate the denom for a(i) calculation
    sigma_denom = np.exp(logsumexp([ ((-1 * dist) / (2 * (tau**2))) for dist in x_l2_distance]))

    distance_weight_array = np.array([(np.exp((-1 * dist) / (2 * (tau**2))) / sigma_denom) for dist in x_l2_distance])

    A = np.zeros((n,n), float)
    np.fill_diagonal(A, distance_weight_array)
    A = A + 1e-8 * np.eye(A.shape[0])

    x_train_transposed = x_train.transpose()

    #solving the equastion from part (a)
    Xt_mult_A = np.matmul(x_train_transposed, A)

    left_side = np.matmul(Xt_mult_A, x_train)
    right_side = np.matmul(Xt_mult_A, y_train)

    # weights
    w = np.linalg.solve(left_side, right_side)

    return np.dot(test_datum.transpose(), w)[0]

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses

#to implement
def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    kf = KFold(n_splits=k)
    avg_losses = np.zeros(taus.shape)
    i = 1
    for train_index, test_index in kf.split(x):
        print('Running fold {} out of 5 ...'.format(i))
        i += 1
        losses = run_on_fold(x[test_index], y[test_index], x[train_index], y[train_index], taus)
        avg_losses += losses


    return (avg_losses / k)
    ## TODO


if __name__ == "__main__":

    """Have not normalized data. Normalization resulted in underflow for tau in [10,1000].
    Could use tau in [0.01, 20] instead if normalizing"""
    #normalizing the data
    #x = general_normalizer(x)
    #y = general_normalizer(y, vector=True)



    # uncommen t to do hold out validation on the 10 first data points
    '''
    # splitting arrays for test and train
    test_target = y[0:10]
    test_data = x[0:10,]

    train_target = y[10:]
    train_data = x[10:,:]
    
    for i, target in enumerate(test_target):
        t = 18.0
        predict = LRLS(test_datum=test_data[i,:].reshape(d,1), x_train=train_data, y_train=train_target, tau=t)
        print("Peredict:  {}".format(predict))
        print("Target: {}".format(target))
        print('<---->')
    '''





    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.plot(taus, losses)
    plt.xlabel('Tau values')
    plt.ylabel('Losses')
    plt.show()
    print("min loss = {}".format(losses.min()))

