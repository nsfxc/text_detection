from numpy import *

def randInitializeWeights(layers):

    num_of_layers = len(layers)
    epsilon = 0.12
        
    Theta = []
    for i in range(num_of_layers-1):
        W = zeros((layers[i+1], layers[i] + 1),dtype = 'float64')
        # ====================== TODO ======================
        # Instructions: Initialize W randomly so that we break the symmetry while
        #               training the neural network.
        #
        W = random.rand(layers[i+1], layers[i] + 1) * 2 * epsilon-epsilon
        Theta.append(W)
                
    return Theta
            
