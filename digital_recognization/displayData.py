from numpy import *
from math import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def displayData(X, example_width = 1):
    #Compute rows, cols
    m, n = X.shape
    example_height = int(math.sqrt(n)) 
    example_width  = example_height
    #Compute number of digits to display
    display_rows = int(math.floor(math.sqrt(m)))
    display_cols = int(math.ceil(m / display_rows))
    
    # Padding between images
    pad = 1

    #Setup black display
    display_array = -ones((pad + display_rows * (example_height+pad), pad + display_cols * (example_width + pad)), float);
    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > (m-1):
                break
            x = pad + j * (example_height + pad) + arange(example_height)
            y = pad + i * (example_width + pad) + arange(example_width)
            
            display_array[x[0]:(x[-1]+1),y[0]:(y[-1]+1)] =  X[curr_ex,:].reshape((example_height, example_width))

            curr_ex = curr_ex + 1
            
        if curr_ex > (m-1):
            break
    plt.figure(1)
    plt.imshow(display_array,cmap = cm.Greys_r)
    plt.axis('off')
    plt.draw()

    plt.show()

