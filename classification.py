# This is a library of classification model implementations.

import numpy as np

# K-nearest neighbours classifier.
# This assumes the features are in a metric space.
# x is some point in the feature space.
# k is the number of neighbours to search for.
# X is the training features, each row is a data point, each column is a feature.
# y are the classes of these features.
# Returns the classes of the k-nearest neighbours.
# Note that any parameters of the metric, such as p in the Lp metric need to be specified,
# e.g. by using def metric(x,y): return lp_metric(x,y,p=p0).
# Also note that there are more efficient algorithms for finding the k smallest values in a list/array.
from mathematics import euclidean_metric
def knns(x, k, X, y, metric = euclidean_metric):
    def dist_x(z):
        return metric(x,z)
    if(len(X.shape)==1):
        X = np.reshape(X, (X.shape[0],1))
    dist = np.apply_along_axis(dist_x, 1, X)
    dist, data, classes = zip(*sorted(zip(dist, X,y)))
    return classes[0:k]

# Classify a point x according to the majority vote of it's k nearest neighbours.
# Ties are broke at random.
def knn_classifier(x,k, X, y, metric = euclidean_metric):
    knn = knns(x,k,X,y,metric)
    unique, counts = np.unique(knn, return_counts=True)
    best = np.argwhere(counts == np.max(counts))
    if len(best)>1:
        i = np.random.choice(best)
    else:
        i = best[0]
    return unique[i]

# Get empirical prior probabities.
def priors(y):
    unique, counts = np.unique(y, return_counts = True)
    prior = {}
    n = len(y)
    for yi, count in zip(unique, counts):
        prior[yi] = count/n
    return prior
