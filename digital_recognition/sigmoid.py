from numpy import *

def sigmoid(z):

    # SIGMOID returns sigmoid function evaluated at z
    g = 1.0/(1.0 + exp(-z))

    return g
