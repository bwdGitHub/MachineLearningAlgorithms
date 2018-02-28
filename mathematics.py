# Mathematics helper functions.

# Simple Lp metric.
def lp_metric(x,y, p=2.0):
    return np.power(np.sum(np.power(np.absolute(x-y), p)),1/p)
def euclidean_metric(x,y):
    return lp_metric(x,y,2)
