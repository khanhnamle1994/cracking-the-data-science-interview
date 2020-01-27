"""
The neural network is trained using backpropagation algorithm. The goal is to learn the XOR function.
One forward and backward pass of a single training example is called iteration. Each itearation has 10 steps.
> Step 0: Weight Initialization

Forward Pass:
> Step 1: Input Layer
> Step 2: Hidden Layer
> Step 3: Output Layer
> Step 4: Calculate the cost

Backpropagation Pass:
> Step 5: Calculate error in the output layer
> Step 6: Calculate error in the hidden layer
> Step 7: Calculate error with respect to weights between hidden and output layer
> Step 8: Calculate error with respect to weights between input and hidden layer
> Step 9: Update the weights between hidden and output layer
> Step 10: Update the weights between input and hidden layer
"""

import numpy as np

x_XOR = np.array([ [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1] ])
y_truth = np.array( [[0], [1], [1], [0]] )

np.random.seed(1)
syn_0 = 2 * np.random.random((3, 4)) - 1
syn_1 = 2 * np.random.random((4, 1)) - 1

def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

def sigmoid_output_to_derivative(output):
    return output * (1 - output)

for iteration in range(25000):
    # Forward Pass
    layer_1 = sigmoid(np.dot(x_XOR, syn_0))
    layer_2 = sigmoid(np.dot(layer_1, syn_1))
    error = layer_2 - y_truth

    # Backpropagation Pass
    layer_2_delta = error * sigmoid_output_to_derivative(layer_2)
    layer_1_error = layer_2_delta.dot(syn_1.T)
    layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

    syn_1 -= layer_1.T.dot(layer_2_delta)
    syn_0 -= x_XOR.T.dot(layer_1_delta)

print("Output after training: \n", layer_2)
