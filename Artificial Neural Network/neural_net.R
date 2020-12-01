######################################################
# ALDA: hw3_neural_net.R 
# Instructor: Dr. Thomas Price
# Mention your team details here
# @author: Yang Shi/yanga
#
# Group Number: G32
#
# Team Members:
#   Shreyas Chikkballapur Muralidhara - (schikkb)
#   Mangalnathan Vijayagopal          - (mvijaya2)
#   Nischal Badarinath Kashyap        - (nkashya)
#
#######################################################
# NOTE: In this homework, neural network will be calculated in matrix calculations rather than scalars.
# The input of the neural is a matrix of size (n,m), where n is the size of the training set, m is the number of attributes.
# All the calculations of derivatives, will therefore be matrix derivative calculations.


sigmoid <- function(x) {
  # Calculate sigmoid of a matrix x
  
  # Inputs:
  # x: a matrix of values
  
  # Output:
  # A matrix with the same size of the input x, where every element x_i is the result of sigmoid(x_i) 
  
  return(1/(1+exp(-x)))
  
}

sigmoid_derivative <- function(x) {
  # Calculate the derivative of sigmoid function with respect to a matrix x.
  
  # Inputs:
  # x: a matrix of values
  
  # Output:
  # A matrix with the same size of the input x, where every element x_i is the result of the derivative of sigmoid(x_i).
  return(x*(1-x))
}

calculate_loss <- function(y_pred, y) {
  # Calculate the loss of predictions and the label.
  
  # Inputs:
  # y_pred: a vector of activations from the last layer of the network.
  # y: a vector of the label of the training samples.
  
  # Output:
  # A number that is the total MSE loss of y_pred and y.
  return(sum((y-y_pred)^2))
}

calculate_activations <- function(input_matrix, weight_matrix) {
  # Calculate the activations of a layer
  
  # Inputs:
  # input_matrix: a matrix, composed of vectors of inputs. The size of the matrix is (n,m), 
  # where n is the number of samples, and m is the number of the attributes, 
  # or the number of hidden units from last layer.
  # weight_matrix: a matrix, containing the weight for a layer. The dimention of the matrix is (m,q),
  # where q is the number of hidden units for this layer.
  
  # Output:
  # A matrix with the size (n,q), activated by the sigmoid function. 
  sigmoid(input_matrix %*% weight_matrix)
}

calculate_dCdw <- function(in_activations, out_activations, out_dCdf) {
  # Calculate the derivative of loss function with respect to a weight matrix w
  
  # Inputs:
  # in_activations: a matrix of the original input of the layer with weight w.
  # out_activations: a matrix of the original output of the layer with the weight w.
  # out_dCdf: The derivative of the loss function to the out_activations.
  
  # Hint 1: in the case of the last layer, out_dCdf would be the derivative of the loss
  # with respect to the activation of the last layer, which is y_pred.
  
  # Hint 2: Remember that dC/dw = dC/df * df/dx * dx/dw, where:
  # C is the cost function, f is the activation's output, x is input to the activation function, and w is a weight
  # Use the derivatives you calcualted in Problem 3 of the homework to help you implement this function
  
  # Output:
  # A matrix with the same size of the target matrx w, recording the derivative of loss to w.
  
  return(t(in_activations) %*% (out_dCdf * sigmoid_derivative(out_activations)))
}

# Note: 522 Only
calculate_dCdf <- function(weight_matrix, out_activations, out_dCdf) {
  # Calculate the derivative of loss function with respect to an activation output of one layer
  
  # Inputs:
  # weight_matrix: a weight matrix for the current layer
  # out_activations: a matrix of the activation values output from this layer.
  # out_dCdf: The derivative of the loss function to the out_activations of this layer.
  
  # Hint 1: This will only be needed in cases of a 2-layer network.
  # Hint 2: Remember that dC/df_{L-1} = dCdf_L/df_L * df_L/dx * dx/df_{L-1}, where:
  # C is the cost function, f_{L-1} is the activation at the previous layer, f_L is the activation at this layer,
  # and x is input to the activation function at this layer
  
  # Output:
  # A matrix with the same size of the activation f, recording dC/df_{L-1}, the derivative of loss to 
  # f, the activations of the previous layer.
  
  return((out_dCdf * sigmoid_derivative(out_activations)) %*% t(weight_matrix))
  
}

neuralnet <- function(x_train, y_train, nodes_layer = 4, n_attributes = 8, learning_rate=0.001, epochs=150) {
  # Implement the neural network.
  
  # Inputs:
  # x_train: The training dataset. A dataframe that has n samples, m attributes.
  # y_train: The labels for training dataset. A dataframe that has n samples, 1 column with the class values.
  # nodes_layer: Integer. In cases of 2-layer neural network, the number of neurons for the first layer is defined here.
  # n_attributes: Integer. Number of attributes.
  # learning_rate: Float. Learning rate of of the neural network.
  # epochs: The number of iterations in training process.
  
  #-------------------------------------------------------------#
  # Data and matrix initialization
  
  # Convert the training dataset to matrix 
  X_train <- data.matrix(x_train)
  Y_train <- data.matrix(y_train)
  
  
  # Random initialize the weight matrix for each layer using rnorm
  layer1_weights <- matrix(rnorm(n_attributes* nodes_layer , mean=0,sd=1),nrow = n_attributes,ncol = nodes_layer)
  
  layer2_weights <- matrix(rnorm(1 * nodes_layer,mean=0,sd=1), nrow = nodes_layer, ncol = 1)
  
  #-------------------------------------------------------------#
  # Training process
  for (i in 1:epochs) {
    #-------------------------------------------------------------#
    # Forward Propagation
    layer1_activations <- calculate_activations(X_train,layer1_weights)
    y_pred <- calculate_activations(layer1_activations,layer2_weights)
    
    #-------------------------------------------------------------#
    # Calculating training loss
    loss <- calculate_loss(y_pred,Y_train)
    
    #-------------------------------------------------------------#
    # Derivative calculation
    
    # Compute the Out_dCdf for layer 2
    out_dCdf_L2 <- (-2) * (Y_train - y_pred)
    
    delta_weight2 <- calculate_dCdw(layer1_activations, y_pred, out_dCdf_L2)
    out_dCdf_L1 <- calculate_dCdf(layer2_weights, y_pred, out_dCdf_L2)
    delta_weight1 <- calculate_dCdw(X_train, layer1_activations, out_dCdf_L1)
    
    
    #-------------------------------------------------------------#
    # Updating weight matrices
    layer1_weights <- layer1_weights - (learning_rate * delta_weight1)
    layer2_weights <- layer2_weights - (learning_rate * delta_weight2)
    
  }
  #-------------------------------------------------------------#
  # Printing the final training loss
  print(loss)
  
}
