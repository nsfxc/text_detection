from numpy import *
from sigmoid import sigmoid

def predict(Theta, X):
    # Takes as input a number of instances and the network learned variables
    # Returns a vector of the predicted labels
    
    # Useful values
    m = X.shape[0]
    num_labels = Theta[-1].shape[0]
    num_layers = len(Theta) + 1

    # ================================ TODO ================================
    # You need to return the following variables correctly
    p = zeros((1,m))
    a = ones(X.shape[0])
    a = vstack((a,X.transpose()))
    for i in range(num_layers-1):
	z = dot(Theta[i],a)
	a = sigmoid(z)
	if i != num_layers-2:
	    a = vstack((ones(a.shape[1]),a))
    h = a.transpose()
    for i in range(m):
	p[0,i] = argmax(h[i]) 	
    
    return p

