from numpy import *
from debugInializeWeights import debugInitializeWeights
from costFunction import costFunction
from unroll_params import unroll_params
from predict import predict

def checkNNCost(lambd):

    input_layer_size  = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m          = 5;
    layers     = [3, 5, 3]
    
    Theta = [] 
    Theta.append(debugInitializeWeights(hidden_layer_size, input_layer_size))
    Theta.append(debugInitializeWeights(num_labels, hidden_layer_size))
    nn_params = unroll_params(Theta)
    
    X = debugInitializeWeights(m, input_layer_size - 1)
    y = remainder(arange(m)+1, num_labels)
 	
    cost = costFunction(nn_params, layers, X, y, num_labels, lambd)
    print 'Cost: ' + str(cost)
