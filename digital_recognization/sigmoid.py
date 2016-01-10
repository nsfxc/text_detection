from numpy import *

def sigmoid(z):

    # SIGMOID returns sigmoid function evaluated at z
    g = zeros(shape(z))
    tmp = zeros(shape(z))
    tmp = fill(1.0)
    g = tmp/(tmp + exp(z))

    return g
