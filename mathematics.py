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

# A one dimensional Gaussian helper function.
def gaussian(x,m,s):
    exponent = ((x-m)**2)/(2*(s**2))
    denominator = s*np.sqrt(2*np.pi)
    return np.exp(-exponent)/denominator

def multivariate_gaussian(x, m, s, dimension):
    inv = np.linalg.inv(s)
    exponent = np.dot(inv, (x-m))
    exponent = np.dot(np.transpose(x-m), exponent)
    exponent/=-2
    print(exponent)
    denominator = np.sqrt(np.power(2*np.pi, dimension)*np.linalg.det(s))
    print(denominator)
    return np.exp(exponent)/denominator
