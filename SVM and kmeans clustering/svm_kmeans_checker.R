#############
# hw4 checker file
# Do not submit this file
# 
#############
# clear workspace
rm(list=ls(all=T))
cat('\014')

# source hw4.R
source('./hw4.R')

# set your working directory
# setwd()

# install all necessary packages
required_packages = c("e1071", "stats", "dplyr", "ggplot2")
for(package in required_packages){
  if(!(package %in% installed.packages())){
    install.packages(package, dependencies = T)
  }   
}


# load the packages
library('e1071') # for SVM
library('stats') # for clustering
library('dplyr') # if needed
library('ggplot2') # for visualization

############################################################################################################
# Helper functions
# TA will use something similar to load data for his own system
# For regression data
load_data <- function(data_folder='./data/', learning_type){
  # this method will read data for clustering/classification and return list containing two data frames:
  
  # for clustering (specified by learning_type = "clustering") 
  # two columns (x, y) of continous data (of type double)
  
  # for classification (specified by learning_type = "classification")
  # first 4 columns (x1-x4) are attributes, last column (class) is your dependent variable (factor))
  
  # for classification, please note, TA WILL use the same training dataset, but a different test set 
  # TA's test set will have the same attributes (x1-x4, class), but may contain different number of data points
  
  # for clustering, please note, TA will use a different dataset
  
  # make sure dependent variable is of type factor if this is classification
  if(learning_type == 'classification'){
    train_df <- read.csv(paste0(data_folder, learning_type, '-train.csv'), header=T)
    test_df <- read.csv(paste0(data_folder, learning_type, '-test.csv'), header=T)
    train_df$class <- as.factor(train_df$class)
    test_df$class <- as.factor(test_df$class)
    return(list(train_df, test_df))
  }else{
    data_df <- read.csv(paste0(data_folder, learning_type, '_sample.csv'), header = T)
  }
  
}

##########################################################################################################
# Load data
# load data necessary for clustering
clustering_data <- load_data(data_folder='./data/', learning_type='clustering')

# load data necessary for classification
clf_data <- load_data(data_folder='./data/', learning_type='classification')
clf_train_df <- clf_data[[1]]
clf_test_df <- clf_data[[2]]

###############################################
# Clustering

# We're giving you the k-means centers, so the seed shouldn't be needed,
# but it's good practice, just in case
set.seed(100)

# KMeans
kmeans_result <- alda_cluster(data_df = clustering_data, n_clusters = 2, clustering_type = "kmeans")
kmeans_sse <- alda_calculate_sse(data_df = clustering_data, cluster_assignments = kmeans_result)
# Should be 204.000760476334
print(paste("Kmeans SSE for given params = ", kmeans_sse))

# Single link
single_link_result <- alda_cluster(data_df = clustering_data, n_clusters = 2, clustering_type = "single-link")
single_link_sse <- alda_calculate_sse(data_df = clustering_data, cluster_assignments = single_link_result)
# Should be 319.693456992432
print(paste("Single link SSE for given params = ", single_link_sse))

# complete link
complete_link_result <- alda_cluster(data_df = clustering_data, n_clusters = 2, clustering_type = "complete-link")
complete_link_sse <- alda_calculate_sse(data_df = clustering_data, cluster_assignments = complete_link_result)
# Should be 221.04197449858
print(paste("Complete link SSE for given params = ", complete_link_sse))


# Setup for analysis section in clustering
# generate the elbow plot for kmeans for c(1, 2, 3, 4, 5, 6, 7)
# It should be saved to disk
alda_kmeans_elbow_plot(data_df = clustering_data, k_values = c(1, 2,3,4,5,6,7))

# Next, lets evaluate visually by visualizing them
plot(clustering_data$x, clustering_data$y, type='p', pch = '*', col=kmeans_result, main = "KMeans with 2 clusters", xlab = 'x', ylab = 'y')
plot(clustering_data$x, clustering_data$y, type='p', pch = '*', col=single_link_result, main = "Single Link with 2 clusters", xlab = 'x', ylab = 'y')
plot(clustering_data$x, clustering_data$y, type='p', pch = '*', col=complete_link_result, main = "Complete link with 2 clusters", xlab = 'x', ylab = 'y')

########################################################
# SVM classification

# Reset the seed
set.seed(100)

# linear kernel
linear_svm_result <- alda_svm(x_train = clf_train_df[,-5], x_test = clf_test_df[,-5], y_train = clf_train_df[,5], 
                              kernel_name = 'linear')
# Print the results: should have 111 support vectors (note this is dependent on the seed)
linear_svm_result

# radial kernel
radial_svm_result <- alda_svm(x_train = clf_train_df[,-5], x_test = clf_test_df[,-5], y_train = clf_train_df[,5], 
                              kernel_name = 'radial')

# sigmoid kernel
sigmoid_svm_result <- alda_svm(x_train = clf_train_df[,-5], x_test = clf_test_df[,-5], y_train = clf_train_df[,5], 
                               kernel_name = 'sigmoid')

# polynomial kernel
polynomial_svm_result <- alda_svm(x_train = clf_train_df[,-5], x_test = clf_test_df[,-5], y_train = clf_train_df[,5], 
                                  kernel_name = 'polynomial')

# plot the results and make comparisons
plot(clf_test_df[,1], clf_test_df[,2], type='p', pch = '*', col=clf_test_df[,5], main = "Ground Truth", xlab = 'x1', ylab = 'x2')
plot(clf_test_df[,1], clf_test_df[,2], type='p', pch = '*', col=linear_svm_result[[2]], main = "SVM linear kernal", xlab = 'x1', ylab = 'x2')
plot(clf_test_df[,1], clf_test_df[,2], type='p', pch = '*', col=radial_svm_result[[2]], main = "SVM radial kernal", xlab = 'x1', ylab = 'x2')
plot(clf_test_df[,1], clf_test_df[,2], type='p', pch = '*', col=sigmoid_svm_result[[2]], main = "SVM sigmoid kernal", xlab = 'x1', ylab = 'x2')
plot(clf_test_df[,1], clf_test_df[,2], type='p', pch = '*', col=polynomial_svm_result[[2]], main = "SVM polynomial kernal", xlab = 'x1', ylab = 'x2')

# compare all classifiers
all_classifier_summary <- classification_compare_accuracy(y_test=clf_test_df[,5], 
                                                          linear_kernel_prediction = linear_svm_result[[2]], 
                                                          radial_kernel_prediction = radial_svm_result[[2]], 
                                                          polynomial_kernel_prediction = polynomial_svm_result[[2]], 
                                                          sigmoid_kernel_prediction = sigmoid_svm_result[[2]])

print(paste('Best classification model =', all_classifier_summary[[1]], 'Overall Accuracy =', all_classifier_summary[[2]]))
# Checker value: the accuracy should be 96.551724137931. (Note this assumes running all code after setting seed above)






