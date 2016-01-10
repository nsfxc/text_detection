from numpy import *
from read_dataset import read_dataset
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from displayData import displayData
from randInitializeWeights import randInitializeWeights
from costFunction import costFunction
from unroll_params import unroll_params
from roll_params import roll_params
from scipy.optimize import *
from predict import predict
from backwards import backwards
from checkNNCost import checkNNCost
from checkNNGradients import checkNNGradients
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient



# ================================ Step 1: Loading and Visualizing Data ================================
print "\nLoading and visualizing Data ...\n"

#Reading of the dataset
# You are free to reduce the number of samples retained for training, in order to reduce the computational cost
size_training = 60000     # number of samples retained for training
size_test     = 10000     # number of samples retained for testing
images_training, labels_training, images_test, labels_test = read_dataset(size_training, size_test)

# Randomly select 100 data points to display
random_instances = range(size_training)
random.shuffle(random_instances)
displayData(images_training[random_instances[0:100],:])

raw_input('Program paused. Press enter to continue!!!')

# ================================ Step 2: Setting up Neural Network Structure &  Initialize NN Parameters ================================
print "\nSetting up Neural Network Structure ...\n"

# Setup the parameters you will use for this exercise
input_layer_size   = 784        # 28x28 Input Images of Digits
num_labels         = 10         # 10 labels, from 0 to 9 (one label for each digit) 

num_of_hidden_layers = int(raw_input('Please select the number of hidden layers: '))
print "\n"

layers = [input_layer_size]
for i in range(num_of_hidden_layers):
    layers = layers +  [int(raw_input('Please select the number nodes for the ' + str(i+1) + ' hidden layers: '))]
layers = layers + [num_labels]

raw_input('\nProgram paused. Press enter to continue!!!')

print "\nInitializing Neural Network Parameters ...\n"

# ================================ TODO ================================
# Fill the randInitializeWeights.py in order to initialize the neural network weights. 
Theta = randInitializeWeights(layers)

# Unroll parameters
nn_weights = unroll_params(Theta)

raw_input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 3: Sigmoid  ================================================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print "\nEvaluating sigmoid function ...\n"

g = sigmoid(array([1, -0.5, 0,  0.5, 1]))
print "Sigmoid evaluated at [1 -0.5 0 0.5 1]:  "
print g

raw_input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 4: Sigmoid Gradient ================================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

print "\nEvaluating Sigmoid Gradient function ...\n"

g = sigmoidGradient(array([1, -0.5, 0,  0.5, 1]))
print "Sigmoid evaluated at [1 -0.5 0 0.5 1]:  "
print g

raw_input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 5: Implement Feedforward (Cost Function) ================================

print "\nChecking Cost Function without Regularization (Feedforward) ...\n"

lambd = 0.0
checkNNCost(lambd)

print 'This value should be about 2.09680198349'

raw_input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 6: Implement Feedforward with Regularization  ================================

print "\nChecking Cost Function with Reguralization ... \n"

lambd = 3.0
checkNNCost(lambd)

print 'This value should be about 2.1433733821'


raw_input('\nProgram paused. Press enter to continue!!!')

# ================================ Step 7: Implement Backpropagation  ================================

print "\nChecking Backpropagation without Regularization ...\n"

lambd = 0.0
checkNNGradients(lambd)
raw_input('\nProgram paused. Press enter to continue!!!')


# ================================ Step 8: Implement Backpropagation with Regularization ================================

print "\nChecking Backpropagation with Regularization ...\n"

lambd = 3.0
checkNNGradients(lambd)

raw_input('\nProgram paused. Press enter to continue!!!')


# ================================ Step 9: Training Neural Networks & Prediction ================================
print "\nTraining Neural Network... \n"

#  You should also try different values of the regularization factor
lambd = 3.0

res = fmin_l_bfgs_b(costFunction, nn_weights, fprime = backwards, args = (layers,  images_training, labels_training, num_labels, 1.0), maxfun = 50, factr = 1., disp = True)
Theta = roll_params(res[0], layers)

raw_input('\nrogram paused. Press enter to continue!!!')

print "\nTesting Neural Network... \n"

pred  = predict(Theta, images_test)
print '\nAccuracy: ' + str(mean(labels_test==pred) * 100)

