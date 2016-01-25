from numpy import *
from debugInializeWeights import debugInitializeWeights
from computeNumericalGradient import computeNumericalGradient
from backwards import backwards
from unroll_params import unroll_params
from costFunction import costFunction

def checkNNGradients(lambd):

    input_layer_size  = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m          = 5;
    layers     = [3, 5, 3]

    # In this point we generate a number of random data
    Theta = [] 
    Theta.append(debugInitializeWeights(hidden_layer_size, input_layer_size))
    Theta.append(debugInitializeWeights(num_labels, hidden_layer_size))

    X = debugInitializeWeights(m, input_layer_size - 1)
    y = remainder(arange(m)+1, num_labels)
    
    # Unroll parameters
    nn_params = unroll_params(Theta)

    # Compute Numerical Gradient
    numgrad = computeNumericalGradient(nn_params,layers, X, y, num_labels, lambd)

    # Compute Analytical Gradient (BackPropagation)
    truegrad = backwards(nn_params, layers, X, y, num_labels, lambd)

    
    print concatenate(([numgrad], [truegrad]), axis = 0).transpose()
    print "The above two columns must be very similar.\n(Left-Numerical Gradient, Right-Analytical Gradient (BackPropagation)\n"
    
    diff = linalg.norm(numgrad - truegrad) / linalg.norm(numgrad + truegrad)
    print "\nNote: If the implementation of the backpropagation is correct, the relative different must be quite small (less that 1e-09)."
    print "Relative difference: " + str(diff) + "\n"
