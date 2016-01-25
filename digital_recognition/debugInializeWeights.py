from numpy import *

def debugInitializeWeights(fan_out, fan_in):
    W = zeros((fan_out, 1 + fan_in))

    W = reshape(sin(arange(size(W))+1.0), W.shape ) / 10. 
    return W
