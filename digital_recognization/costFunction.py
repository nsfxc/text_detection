from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def costFunction(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the cost function of the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Unroll Params
    Theta = roll_params(nn_weights, layers)
    
    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a 
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((num_labels, m))
    for i in range(m):
	yv[y[i],i] = 1

    # In this point calculate the cost of the neural network (feedforward)
    # a: the result obtained after each layer
    a = ones(X.shape[0])
    a = vstack((a,X.transpose()))
    for i in range(num_layers-1):
	z = dot(Theta[i],a)
	a = sigmoid(z)
	if i != num_layers-2:
	    a = vstack((ones(a.shape[1]),a))
    #h: final result
    h = a.transpose()
	
    #calculate of the cost J
    J = 0
    for i in range(m):
	for k in range(num_labels):
	   J = J + (-yv[k,i] * log(h[i][k]) - (1-yv[k,i]) * log(1.0 - h[i][k]))
    J = J/m;
    
    #regularization
    tmp = 0
    for i in range(num_layers-1):
    	for j in range(Theta[i].shape[0]):
	    for k in range(1,Theta[i].shape[1]):
		tmp = tmp + Theta[i][j][k] * Theta[i][j][k]
    J = J + tmp * lambd/(2.0*m) 
    
    return J

    

