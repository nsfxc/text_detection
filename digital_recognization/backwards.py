from numpy import *
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient
from roll_params import roll_params
from unroll_params import unroll_params

def backwards(nn_weights, layers, X, y, num_labels, lambd):
    # Computes the gradient fo the neural network.
    # nn_weights: Neural network parameters (vector)
    # layers: a list with the number of units per layer.
    # X: a matrix where every row is a training example for a handwritten digit image
    # y: a vector with the labels of each instance
    # num_labels: the number of units in the output layer
    # lambd: regularization factor
    
    # Setup some useful variables
    m = X.shape[0]
    num_layers = len(layers)

    # Roll Params
    # The parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    Theta = roll_params(nn_weights, layers)
  
    # You need to return the following variables correctly 
    Theta_grad = [zeros(w.shape) for w in Theta]

    # The vector y passed into the function is a vector of labels
    # containing values from 1..K. You need to map this vector into a 
    # binary vector of 1's and 0's to be used with the neural network
    # cost function.
    yv = zeros((num_labels, m))
    for i in range(m):
	yv[y[i],i] = 1

    # In this point implement the backpropagaition algorithm 
    A = []
    a = ones(X.shape[0])
    a = vstack((a,X.transpose()))
    Z = []
    Z.append(a)
    for i in range(num_layers-1):
        A.append(a.transpose())
	z = dot(Theta[i],a)
	Z.append(z)
	a = sigmoid(z)
	if i != num_layers-2:
	    a = vstack((ones(a.shape[1]),a))  
  
    # A: list of result after each layer
    A.append(a.transpose())
    h = a.transpose()

    # delta for the last layer
    delta = h - yv.transpose()
    # calculate of gradients
    for j in range(num_layers-2,0,-1):
	Theta_grad[j] = Theta_grad[j] + dot(delta.transpose(),A[j])
	# calculate of delta for current layer(have to remove the first column of Theta)
	tmp = dot(Theta[j][:,1:].transpose(),delta.transpose())
	tmp = tmp.transpose()
	tmp_matrix = zeros(tmp.shape)
	for i in range(m):
	    tmp_matrix[i] = sigmoidGradient(Z[j].transpose()[i])
	delta = tmp_matrix * tmp
    Theta_grad[0] = Theta_grad[0] + dot(delta.transpose(),A[0])

    
    # regularization
    for i in range(num_layers-1):
	for j in range((Theta_grad[i].shape)[0]):
	    for k in range((Theta_grad[i].shape)[1]):
		Theta_grad[i][j,k] = Theta_grad[i][j,k]/m
		if k >=1:
			Theta_grad[i][j,k] = Theta_grad[i][j,k] + lambd/m*Theta[i][j,k]
    # Unroll Params
    Theta_grad = unroll_params(Theta_grad)

    return Theta_grad

    
