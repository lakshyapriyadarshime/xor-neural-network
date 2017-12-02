''' 
	Project - Neural network that implements XOR Gate
	Lakshya Priyadarshi : official.18lakshya@gmail.com		
	B.Tech 2nd Semester, Computer Science & Engineering

	A multi-layered neural network that models the XOR logic gate
	Sigmoid activation function, batch gradient descent algorithm
	Number of layers = 4, Learning rate = 0.1, Epoch = 50000

 '''

#Import all the relevant library and packages
import sys, scipy, numpy as np

# Initial input binary values to boolean function Exclusive-OR
input_vector_x=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])

# Boolean functional value on corresponding binary inputs
output_vector_y=np.array([[1],[1],[0]])

# Define sigmoid activation function to upscale threshold for input x
def sigmoid_activation_function(x):
    return 1/(1 + np.exp(-x))

# Derivative of sigmoid activation function with respect to input x
def derivative_sigmoid_activation_function(x):
    return x * (1 - x)

# Define number of process-cycles / epoch
number_of_process_cycles=50000

# Define learing rate for model training
learning_rate=0.1

# Setting parameters for input layer neurons 
inputlayer_neurons = input_vector_x.shape[1]

# Setting parameters for hidden layer neurons
hiddenlayer_neurons = 4

# Setting parameters for output layer neurons
output_neurons = 1 

# Weight matrix to the hidden layer
weight_matrix_hidden_layer=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))

# Bias matrix to the hidden layer
bias_matrix_hidden_layer=np.random.uniform(size=(1,hiddenlayer_neurons))

# Weight matrix to the outer layer
weight_matrix_outer_layer=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))

# Bias matrix to the outer layer
bias_matrix_outer_layer=np.random.uniform(size=(1,output_neurons))

# Iterate for proces cycle
for i in range(number_of_process_cycles):

    ''' FORWARD PTOPOGATION '''

    # Matrix dot-product of input matrix with weight matrix
    # Linear transformation applied
    hidden_layer_input1=np.dot(input_vector_x,weight_matrix_hidden_layer)
    hidden_layer_input=hidden_layer_input1 + bias_matrix_hidden_layer

    # Non-linear transformation over hidden layers
    hiddenlayer_activations = sigmoid_activation_function(hidden_layer_input)

    # Matrix dot-product of output matrix with weight matrix
    # Linear transformation applied    
    output_layer_input1=np.dot(hiddenlayer_activations,weight_matrix_outer_layer)
    output_layer_input= output_layer_input1+ bias_matrix_outer_layer
    output_by_transformation = sigmoid_activation_function(output_layer_input)
    

    ''' BACKWARD PROPOGATION '''

    # Calculation of error with respect to desired output
    E = output_vector_y - output_by_transformation
    
    # Compute gradient of output layer and hidden layer
    slope_output_layer = derivative_sigmoid_activation_function(output_by_transformation)
    slope_hidden_layer = derivative_sigmoid_activation_function(hiddenlayer_activations)
    
    # Change factor delta 
    delta_output = E * slope_output_layer
    Error_at_hidden_layer = delta_output.dot(weight_matrix_outer_layer.T)
    
    delta_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer
    
    # Batch gradient descent applied 
    weight_matrix_outer_layer = weight_matrix_outer_layer + hiddenlayer_activations.T.dot(delta_output) * learning_rate
    bias_matrix_outer_layer = bias_matrix_outer_layer + np.sum(delta_output, axis=0,keepdims=True) * learning_rate
    weight_matrix_hidden_layer = weight_matrix_hidden_layer + input_vector_x.T.dot(delta_hiddenlayer) * learning_rate
    bias_matrix_hidden_layer = bias_matrix_hidden_layer + np.sum(delta_hiddenlayer, axis=0, keepdims=True) * learning_rate

print(output_by_transformation)





