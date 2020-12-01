## Problem Description

i. **Dataset description:** data/pima-indians-diabetes.csv. You will use it to test a neural network that you write from scratch.  

ii. **Instructions:** You will implement a neural network from scratch. The framework code, detailed instructions and the allowed packages are in provided in hw3 neural net.R, 
and the corresponding checker code is in hw3 neural net checker.R. As in other coding questions, you should only submit hw3 neural net.R as the answer for this question. 
you will implement a 2-layer neural network.   

iii. **Note 1:** As explained in the code comments, you will be implementing this ANN using matrix multiplication. Rather than doing backpropagation one instance at a 
time , you will calculating the weight updates for all training instances at once, and summing these to calculate the net change. In R, the %*% operator performs matrix 
multiplication.    

iv. **Note 2:** For a given layer in the ANN, we use a matrix W to hold the values of all weights, where Wi;j is the weight of the edge going from input node i to output node j.

v. **Sigmoid implementation :** You will write code in the function sigmoid() to Calculate the sigmoid function (f(x) = 1+e􀀀x ) given an input vector x. Correspondingly, you 
will need to also write code to calculate the derivatives of sigmoid function in sigmoid derivative(), which takes an input vector x, and calculate delta(f)/delta(x).  

vi. **Loss Calculation :** Assume that our loss function is C = summation(yi 􀀀 Oi)^2 over all the instances i, where yi is the class label for the instance i, and Oi is 
the activation value of the last layer for that instance. You will implement the code to calculate this function in calculate loss(). Activation Calculation (0 pts): In the function calculate activations(), 
we have implemented a function to calculate the activation vector for a given node, given the input matrix of activations from the prior layer, and the weight matrix. The input is a matrix, where each
row corresponds to one training instance and each column corresponds to the activation of a node in the previous layer. Your output is a vector, where each item in the vector represents
the outputs for one training instance. Goal: Read over this function and test it until you understand how the matrix multiplication can calculate a full layer of the ANN for all training
instances in one matrix calculation.

vii. **Gradient Calculation:** There are two kinds of gradients you will need to calculate to perform back propagation, besides the gradient for the activation
function that you have already calculated in sigmoid derivative(). In calculate dCdw(), you will need to calculate the derivative of the cost with respect to a weight matrix. 522
only: In calculate dCdf(), you will need to calculate the derivative of the cost with respect to activation values, calculated by calculate activations(). Hint: Use https:
//en.wikipedia.org/wiki/Matrix_calculus for references about derivative calculation for matrices.

viii. **Network Training :** After writing these functions above, you have all the components you need to train a neural network. You will implement your
neural network in the function neuralnet. The first step is to read the training dataset in, and use rnorm to randomly initialize the weight matrices. You need one matrix for each layer in the
ANN. The number of nodes in each layer is given as a function parameter.After initialization, you will need to use a for loop to train this neural network for the given
number of iterations, or epochs, also given as a parameters. In every iteration of the loop, you will first do a forward pass to calculate the activations at your 
first hidden layer and your output layer, using your calculate activations() function. Then use the function calculate loss() to get the loss (error) C for this iteration.
You will then use the loss to calculate the gradient of the cost with respect to the weights, using calculate dCdw() (for the final layer) and calculate dCdf() (for the first layer { 522
only). Note that you may need to use these functions multiple times if you are implementing a multiple layer neural network. 

The final step is to use the gradient you calculated for every  layer's weight to update the weights. You can print the loss for every epochs to track the change of the loss.