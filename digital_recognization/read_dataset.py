from mnist2 import load_mnist
from numpy import *

def read_dataset(size_training, size_testing):
    digits = [0,1,2,3,4,5,6,7,8,9]
    images_train, labels_train = load_mnist('training',digits)
    images_test, labels_test = load_mnist('testing',digits)

    total_training = len(labels_train)
    if(size_training > total_training):
        size_training = total_training
    total_testing = len(labels_test)
    if(size_testing > total_testing):
        size_testing = total_testing

    random_training_instances = range(total_training)
    random.shuffle(random_training_instances)
    random_testing_instances = range(total_testing)
    random.shuffle(random_training_instances)
   
    images_train = images_train.astype(float64)
    images_test  = images_test.astype(float64)
    images_train = images_train[random_training_instances[0:(size_training)],:]
    labels_train = labels_train[random_training_instances[0:(size_training)]]
    images_test  = images_test[random_testing_instances[0:(size_testing)],:]
    labels_test  = labels_test[random_testing_instances[0:(size_testing)]]

    return images_train, labels_train, images_test, labels_test
