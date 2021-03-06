# This is a library of regression model implementations.

import numpy as np

# Ridge regression - i.e. L2 regularised linear regression
# X_ are features, y are targets. Both should be numpy arrays.
# C is the regularisation constant.
# If X_ already contains a bias column (a column of 1s) you can set add_bias to False.
# Note X_ must be shaped so that the columns are features, and the rows are samples of the data.
# This function relies on numpy for linear algebra.
def ridge_regression(X,y, C=0, add_bias = True):
    if(C<0):
        raise ValueError("Regularisation must be non-negative.")
    if(add_bias):
        X_ = np.ones((X.shape[0], X.shape[1]+1))
        X_[:,:-1] = X
    else:
        X_ = X
    Xt = np.matrix.transpose(X_)
    XtX = np.dot(Xt,X_)
    I = np.identity(XtX.shape[0])
    regularised = XtX + C*I
    try:
        inverse = np.linalg.inv(regularised)
    except np.linalg.LinAlgError as err:
        if("Singular matrix" in str(err)):
            print("Product of feature matrix and its transpose is non-invertible.")
            print("Try setting regularisation to a non-zero value.")
            return
        else:
            raise
    inverseXt = np.dot(inverse, Xt)
    return np.dot(inverseXt, y)

# Lp regularised regression. With p = 1 this is known as Lasso.
# Note this relies on scipy's optimize.minimize function.
# This optimisation could be achieved with gradient descent.
# Note this only works with 1-dimensional targets.
# It is simple to extend to n-dimensional targets by applying this on each dimension.
# i.e. To predict y = (y1,y2) as a linear function of X, predict y1 and y2 individually.
def lp_regression(X,y, C = 1.0, p = 1.0, add_bias = True, max_iter = 1000):
    if(C<0):
        raise ValueError("Regularisation must be non-negative.")
    if(add_bias):
        X_ = np.ones((X.shape[0], X.shape[1]+1))
        X_[:,:-1] = X
    else:
        X_ = X
    Xt = np.transpose(X_)
    XtX = np.dot(Xt,X)
    Xty = np.dot(Xt,y)

    W = np.random.normal(size = (X_.shape[1], 1))
    lpnorm = np.power(np.sum(np.power(np.absolute(W),p)),1/p)
    def loss(W):
        return np.linalg.norm(y - np.dot(X_,W)) + C * lpnorm
    from scipy.optimize import minimize
    return minimize(loss, W, options = {'maxiter':max_iter}).x

# K-nearest neighbours regression.
# Take the average of the k nearest neighbours as a prediction.
# This can also be implemented with weighting, e.g. based on a kernel.
# This would require knns to return the distances between neighbours.
# Note this is defined to only regress one point at a time.
# To regress multiple points you can simply iterate this function.
from mathematics import euclidean_metric
def knn_regression(x,k, X, y, metric = euclidean_metric):
    from classification import knns
    knn = knns(x,k, X, y, metric = metric)
    return np.mean(knn)
