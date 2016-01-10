from numpy import *
from sigmoid import sigmoid

def sigmoidGradient(z):

    # SIGMOIDGRADIENT returns the gradient of the sigmoid function evaluated at z


    g = zeros(z.shape)
    g = sigmoid(z).transpose() * (1.0 - sigmoid(z))
    
    return g




