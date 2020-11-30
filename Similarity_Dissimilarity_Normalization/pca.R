###################################################
# Automated Learning and Data Analysis - HW1
# Instructor: Dr. Thomas Price
# Mention your team details here
#
# Group Number: G32
#
# Team Members:
#   Shreyas Chikkballapur Muralidhara - (schikkb)
#   Mangalnathan Vijayagopal          - (mvijaya2)
#   Nischal Badarinath Kashyap        - (nkashya)
#
###################################################


# You may use the following librarie(s):
require(plyr)
# If you get an error when running these lines, 
# make sure to install the respective libraries

# 1a) read data matrix
read_data <- function(path = "./iris.csv") {
  # Note 1: DO NOT change the function arguments.
  # Input: path: type: string, output: a matrix containing data from iris.csv
  # Write code here to read the csv file as a data frame and return it.
  
  return(read.csv(path, sep = ","))
}

# Part 1: Distance Measurement
calculate_euclidean <- function(p, q) {
  # Input: p, q are numeric vectors of the same length
  # output: a single value of type double, containing the euclidean distance between p and q.
  
  euclidean_dist <- sqrt(sum((p-q)^2))
  return(euclidean_dist)
}

calculate_cosine <- function(p, q) {
  # Input: p, q are numeric vectors of the same length
  # output: a single value of type double, containing the cosine distance between p and q.
  
  cosine_dist <- 1 - ( sum(p*q)/( sqrt(sum(p^2)) * sqrt(sum(q^2)) ) )
  return(cosine_dist)
}

calculate_l_inf <- function(p, q) {
  # Input: p, q are numeric vectors of the same length
  # output: a single value of type double, containing the l_inf distance between p and q.

  l_inf_val <- max(sqrt((p-q)^2))
  return(l_inf_val)
}

# Part 2: principal Component Analysis
principal_component_analysis <- function(data, n){
  # Input: data: the Iris dataframe, with 4 numeric attributes and a 5th nominal class variable
  #        n: the number of the principle component to calculate (e.g. 1 for first principal component)
  # output: a 1 x 4 vector of type double, containing the weights (eigenvector) of the 
  # nth principal component of the dataset.
  
  return(prcomp(data[,-ncol(data)],rank=n))
}

principal_component_calculation <- function(p, component_weights){
  # Input: p is a numeric vector of of length n, e.g. representing a row from the original dataset.
  #        component_weights is a vector length n, containing the weights of a principal component
  #        (e.g. the output from running principal_component_analysis)
  # Output: a single value of type double, containing the first principal component value of the sample.
  
  fpc_value <- unlist(p) * unlist(component_weights[2])
  return(sum(fpc_value))
}

pc1_distance <- function(p, q, component_weights) {
  # Input: p, q are numeric vectors of of length n, e.g. representing rows from the original dataset.
  #        component_weights is a vector length n, containing the weights of a principal component
  #        (e.g. the output from running principal_component_analysis)
  # output: a single value of type double, containing the distance between p and q, projected onto 
  # the first principal component (i.e. |PC1_p - PC1_q|)
    
  PC1_p <- principal_component_calculation(p, component_weights)
  PC1_q <- principal_component_calculation(q, component_weights)
  return(abs(PC1_p-PC1_q))
}
