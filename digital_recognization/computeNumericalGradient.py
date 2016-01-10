from numpy import *
from costFunction import costFunction

def computeNumericalGradient(theta, layers, X, y, num_labels, l):
    numgrad = zeros(theta.shape)
    perturb = zeros(theta.shape)
    e = 0.0001

    for i in range(theta.size):
        perturb[i]  = e;
        loss1 = costFunction(theta - perturb,layers, X, y, num_labels, l)
        loss2 = costFunction(theta + perturb,layers, X, y, num_labels, l)

        # Compute Numerical Gradient
        numgrad[i] = (loss2 - loss1) / (2*e)
        perturb[i] = 0.0

    return numgrad
