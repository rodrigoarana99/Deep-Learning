# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:46:51 2023

@author: HP
"""

import torch
def activation(x):
    """ Sigmoid activation function 
    
        Arguments
        ---------
        x: torch.Tensor
    """
    return 1/(1+torch.exp(-x))

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 5 random normal variables
features = torch.randn((1, 5))
# True weights for our data, random normal variables again
weights = torch.randn_like(features)
# and a true bias term
bias = torch.randn((1, 1))

#features = torch.randn((1, 5)) creates a tensor with shape (1, 5), one row and five columns, that contains values randomly distributed according to the normal distribution with a mean of zero and standard deviation of one
#weights = torch.randn_like(features) creates another tensor with the same shape as features, again containing values from a normal distribution.
#bias = torch.randn((1, 1)) creates a single value from a normal distribution.
#Exercise: Calculate the output of the network with input features features, weights weights, and bias bias. Similar to Numpy, PyTorch has a torch.sum() function, as well as a .sum() method on tensors, for taking sums. Use the function activation defined above as the activation function.
y=activation(torch.sum(features*weights)+bias)
y= activation((features * weights).sum() + bias)
#Exercise: Calculate the output of our little network using matrix multiplication.
## Solution

y = activation(torch.mm(features, weights.view(5,1)) + bias)

### Generate some data
torch.manual_seed(7) # Set the random seed so things are predictable

# Features are 3 random normal variables
features = torch.randn((1, 3))

# Define the size of each layer in our network
n_input = features.shape[1]     # Number of input units, must match number of input features
n_hidden = 2                    # Number of hidden units 
n_output = 1                    # Number of output units

# Weights for inputs to hidden layer
W1 = torch.randn(n_input, n_hidden)
# Weights for hidden layer to output layer
W2 = torch.randn(n_hidden, n_output)

# and bias terms for hidden and output layers
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))

#Exercise: Calculate the output for this multi-layer network using the weights W1 & W2, and the biases, B1 & B2.

### Solution

h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)
print(output)
#o create a tensor from a Numpy array, use torch.from_numpy(). To convert a tensor to a Numpy array, use the .numpy() method
import numpy as np
a = np.random.rand(4,3)
a
b = torch.from_numpy(a)
b
b.numpy()
# Multiply PyTorch Tensor by 2, in place
b.mul_(2)
# Numpy array matches new values from Tensor
a