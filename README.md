### Machine Learning Algorithms
Implementations of some machine learning algorithms as a means for me to get more familiar with how they work.

The algorithms aren't meant to be the most efficient, but are hopefully sufficiently transparent that it's not too hard to see how they work.
Mostly I've tried to only use numpy, however I have used scipy's optimise for minimizing the loss in Lp-regularised regression (this could be replaced by approximating the gradient of this loss, or computing it by hand).

There are 3 files, mathematics.py, regression.py and classification.py. 

Mathematics is a collection of math helper functions.

Regression is a collection of regression methods (just ridge regression, Lp regression and K nearest neighbours regression at the time of writing).

Classification is a collection of classification methods, such as k-nearest neighbours classifier, linear classifiers trained by gradient descent (perceptron), and a so-called plug in classifier.

Much of the theory for these models came from notes I took on the ColumbiaX Machine learning course on edx.org.
