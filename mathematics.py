# Mathematics helper functions.
# Note many of these are possibly defined in numpy already.
import numpy as np
# Simple Lp metric.
def lp_metric(x,y, p=2.0):
    return np.power(np.sum(np.power(np.absolute(x-y), p)),1/p)
def euclidean_metric(x,y):
    return lp_metric(x,y,2)

# A gaussian kernel on Lp metrics.
def gaussian_kernel(x,y,sigma, p=2.0):
    dist = lp_metric(x,y, p=p)
    return np.exp(-(dist**2)/(2*sigma**2))
