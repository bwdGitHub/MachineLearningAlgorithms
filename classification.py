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

# Get mean value of features for each class.
def class_conditional_mean(X,y):
    unique, counts = np.unique(y, return_counts = True)
    means = {}
    for yi, count in zip(unique, counts):
        means[yi] = np.mean(X[y==yi], axis = 0)
    return means

# Get standard deviation for each class.
def class_conditional_std(X,y):
    unique, counts = np.unique(y, return_counts = True)
    stds = {}
    means = class_conditional_mean(X,y)
    for yi, count in zip(unique, counts):
        Xs = X[y ==yi]
        outers = np.zeros(np.outer(Xs[0],Xs[0]).shape)
        for x in Xs:
            outers += np.outer(x-means[yi], x-means[yi])
        stds[yi] = outers/count
    return stds

# We can use the above to find the probabilities that a point is in a particular class, assuming these are Gaussian.
def gaussian2(x,m,s):
    exponent = (np.sum(np.power(x-m,2))/(2*(s**2)))
    denominator = s*np.sqrt(2*np.pi)
    return np.exp(-exponent)/denominator
def plug_in_probability(x, X,y):
    prior = priors(y)
    means = class_conditional_mean(X,y)
    stds = class_conditional_std(X,y)
    prob = {}
    total_prob = 0
    from mathematics import gaussian
    for clf in prior:
        prob[clf] = prior[clf]*gaussian2(x,means[clf],stds[clf])
        #gaussian(x, means[clf], stds[clf])*prior[clf]
        total_prob+=prob[clf]
    # normalise the probabilites
    for clf in prob:
        prob[clf] /= total_prob
    return prob
