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
def plug_in_probability(x, X,y):
    prior = priors(y)
    means = class_conditional_mean(X,y)
    stds = class_conditional_std(X,y)
    prob = {}
    total_prob = 0
    from mathematics import multivariate_gaussian
    for clf in prior:
        prob[clf] = prior[clf]*multivariate_gaussian(x,means[clf],stds[clf])
        total_prob+=prob[clf]
    # normalise the probabilites
    for clf in prob:
        prob[clf] /= total_prob
    return prob

# Linear binary classifier.
# X is input features. W is a weight matrix.
# Returns 1 or -1 as the classes.
# This needs to be trained to find W.
def linear_classifier(X,W, bias=0):
    model = np.dot(X,W) + bias
    pred = np.ones(model.shape)
    pred[model<0] = -1
    return np.transpose(pred)

# Gradient of the loss function that counts misclassifications according to how badly they are wrong.
# i.e. the gradient for gradient descent in perceptron.
# Note there seems to be some awkwardness here with getting misclf_y to be the right shape.
def misclassification_gradient(X,y, W,  bias=0):
    clf = linear_classifier(X,W, bias)
    n = len(clf)
    W_grad = np.zeros(len(W))
    bias_grad = 0
    rows = y!=clf[0]
    misclf_X = X[rows,:]
    misclf_y = y[rows]
    bias_grad = -np.sum(misclf_y)
    W_grad = -np.sum(misclf_X *np.reshape(misclf_y, (len(misclf_y), 1)), axis=0)
    return np.reshape(W_grad, (len(W_grad),1)), bias_grad

# Gradient descent for the misclassification gradient to train a linear classifier.
def misclassification_gradient_descent(X,y, initial_W = None, initial_bias = 0, learning_rate = 1.0, max_iter = 100, verbose = False, steps_per_message = 10):
    bias = initial_bias
    if(initial_W is None):
        if(len(X.shape)==1):
            rows = 1
        else:
            rows = X.shape[1]
        W = np.random.normal(size = (rows,1))
    else:
        W = initial_W
    step = 0
    clf = linear_classifier(X,W,bias)
    num_errors = np.sum(y!=(clf[0]))
    while(step < max_iter and num_errors>0):
        step+=1
        W_grad, bias_grad = misclassification_gradient(X,y,W, bias)
        W = W - learning_rate * W_grad
        bias = bias-learning_rate*bias_grad
        clf = linear_classifier(X,W,bias)
        num_errors = np.sum(y!=(clf[0]))
        if(verbose and step % steps_per_message==0):
            print("There are {} errors at step {}".format(num_errors, step))
    return W, bias
