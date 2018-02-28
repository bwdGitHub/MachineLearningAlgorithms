# This is a library of classification model implementations.

import numpy as np

# K-nearest neighbours classifier.
# This assumes the features are in a metric space.
# x is some point in the feature space.
# k is the number of neighbours to search for.
# X is the training features, each row is a data point, each column is a feature.
# y are the classes of these features.
# Returns the classes of the k-nearest neighbours.
from mathematics import euclidean_metric
def knns(x, k, X, y, metric = euclidean_metric):
    def dist_x(z):
        return metric(x,z)
    dist = np.apply_along_axis(dist_x, 1, X)
    dist, data, classes = zip(*sorted(zip(dist, X,y)))
    return classes[0:k]
