########
# HW4 
# Instructor: Dr. Thomas Price
# Specify your team details here
# 
#
#
#
#########
library(dplyr)
library(ggplot2)

alda_calculate_sse <- function(data_df, cluster_assignments){
  # Calculate overall SSE
  # This code has already been given to you
  # Input:
    # data_df: data frame that has been given to you by the TA (x,y attributes)
    # cluster_assignments: a vector of cluster assignments (e.g. 1, 2), each corresponding to a row in the dataframe,
    #                      which have been generated using any of the clustering algorithms
  # Output:
    # A single value of type double, which is the total SSE of the clusters, using Euclidean distance.
    # To calculate the SSE, first calculate the centroid for the cluster, then calculate to total Euclidean
    # distance (error) from each point to that centroid.

  all_data <- data.frame(data_df, cluster_assignments)
  all_data <- all_data %>%
    group_by(cluster_assignments) %>%
    summarize(within_sse=sum((x - mean(x))^2 + (y - mean(y))^2)) %>%
    summarize(total_sse=sum(within_sse))
  return(all_data$total_sse)
}



alda_kmeans_elbow_plot <- function(data_df, k_values){
  # ~ 8-10 lines
  # Input:
    # data_df: Original data frame supplied to you by the TA
    # k_values: A vector of values of k for k means clustering
  
  # General Information:
    # Run k means for all the values specified in k_values, generate elbow plot
    # Use alda_cluster with kmeans as your clustering type
    # (you can see an example this function call in hw4_checker.R for k = 2, now repeat it for all k_values)
  
  # Output:
    # Nothing, simply generate a plot and save it to disk as "GroupNumber_elbow.png"

  cluster_data <- mapply(alda_cluster, n_clusters = k_values, MoreArgs = list(data_df = data_df, clustering_type = "kmeans"), SIMPLIFY=FALSE)
  cluster_sse <- mapply(alda_calculate_sse, cluster_assignments = cluster_data, MoreArgs = list(data_df = data_df))
  plot(k_values, cluster_sse, type='b', main = "Kmeans elbow plot", xlab = 'K Values', ylab = 'SSE')
  #plot.save() 
  
}


alda_cluster <- function(data_df, n_clusters, clustering_type){
  cluster_center <- matrix(c(1,1,-1,-1), ncol=2, byrow=TRUE) # We provide this as the initial points for kmeans.
  # Perform specified clustering
  
  # Inputs:
  # data_df: The dataset provided to you, 2-dimensional (x1,x2)
  # n_clusters: number of clusters to be created, in the case of kmeans it is the starting centers for the clusters.
  # clustering_type: can be one of "kmeans" or "single-link" or "complete-link"
  
  # Outputs:
  # Cluster assignments for the specified method (vector, with length = nrow(data_df) and values ranging from 1 to n_clusters)
  if(clustering_type == "kmeans"){
    # ~ 1-2 lines
    # allowed packages for kmeans: R-base, stats, dplyr
    # set the max number of iterations to 100, number of random restarts = 1 (let's not break the TA's computer! )
    # choose "Lloyd" as the algorithm 
    
    clusters <- kmeans(data_df, n_clusters, iter.max = 100, nstart = 1, algorithm = "Lloyd")
    return(clusters$cluster)
    
  }else if(clustering_type == "single-link"){
    # ~ 3-5 lines
    # Allowed packages for single-link: R-base, stats, dplyr
    # Use euclidean distance for distance calculation (Hint: Look at dist method from stats package)
    # Note 1: Can you use the data_df directly for hclust, or do you need to compute something first?
            # What does 'd' mean in hclust? 
    # Note 2: Does hclust return the clusters assignments directly, or does it return a dendrogram? 
            # Hint 2: Look up the stats package for a method to cut the tree at n_clusters
            # Visualize the dendrogram - paste this dendrogram in your PDF 
    
    d <- dist(data_df, method = 'euclidean')
    hclust_avg <- hclust(d, method = 'single')
    cut_avg <- cutree(hclust_avg, k = n_clusters)
    return(cut_avg)
    
    
  }else{ #complete link clustering is default
    # ~ 3-5 lines
    # Allowed packages for single-link: R-base, stats, dplyr
    # Use euclidean distance for distance calculation (Hint: Look at dist method from stats package)
    # Note 1: Can you use the data_df directly for hclust, or do you need to compute something first?
    # What does 'd' mean in hclust? 
    # Note 2: Does hclust return the clusters assignments directly, or does it return a dendrogram? 
    # Hint 2: Look up the stats package for a method to cut the dendrogram at n_clusters
    # Visualize the dendrogram - paste this dendrogram in your PDF 
    
    d <- dist(data_df, method = 'euclidean')
    hclust_avg <- hclust(d, method = 'complete')
    cut_avg <- cutree(hclust_avg, k = n_clusters)
    return(cut_avg)
    
      
  }
}



alda_svm <- function(x_train, x_test, y_train, kernel_name){
  # Perform classification using support vector machines (linear/radial/sigmoid)
  
  # Inputs:
  # x_train: training data frame(4 variables, x1-x4)
  # x_test: test data frame(4 variables, x1-x4)
  # y_train: dependent variable, training data (factor)
  # kernel_name: specifies type of SVM kernel, string variable, can be of type 'linear', 'radial' or 'sigmoid' or 'polynomial'
  
  # General information
  # Both training data and test data have already been scaled - so you don't need to scale it once again.
  
  # Kernel specific information: using 10-fold cross-validation, perform hyperparameter tuning for each kernel as shown below:
  # Linear: 
  # 'cost' parameter: for the following values: c(0.01, 0.1, 1, 10)
  # radial: 
  # 'cost' parameter: for the following values: c(0.01, 0.1, 1, 10), 
  # 'gamma' parameter: for the following values: c(0.05, 0.5, 1, 2)
  # polynomial:
  # 'cost' parameter: for the following values: c(0.01, 0.1, 1, 10), 
  # 'gamma' parameter: for the following values: c(0.05, 0.5, 1, 2)
  # 'degree' parameter: for the following values: c(1,2,3)
  # sigmoid:
  # 'cost' parameter: for the following values: c(0.01, 0.1, 1, 10), 
  # 'gamma' parameter: for the following values: c(0.05, 0.5, 1, 2)
  
  # Output:
  # A list with two elements, first element = model generated, second element = predictions on test data (factor) 
  
  # Word of caution:
  # Make sure that you pick the best parameters after tuning
  
  # allowed packages: R-base, e1071
  
  
  if(kernel_name == "radial"){
    # ~1-2 lines 
    tuned_model <- tune.svm(x_train,y_train,kernel=kernel_name, cost=c(0.01, 0.1, 1, 10), gamma = c(0.05, 0.5, 1, 2))

  }else if(kernel_name == 'polynomial'){
    #~1-2 lines
    tuned_model <- tune.svm(x_train,y_train,kernel=kernel_name, cost=c(0.01, 0.1, 1, 10), gamma = c(0.05, 0.5, 1, 2), degree=c(1,2,3))

  }else if(kernel_name == 'sigmoid'){
    #~1-2 lines
    tuned_model <- tune.svm(x_train,y_train,kernel=kernel_name, cost=c(0.01, 0.1, 1, 10), gamma = c(0.05, 0.5, 1, 2))

  }else{ # default linear kernel
    #~1-2 lines
    tuned_model <- tune.svm(x_train,y_train,kernel=kernel_name, cost=c(0.01, 0.1, 1, 10))

  }
  svm_best <-tuned_model$best.model
  pred <- predict(svm_best,x_test)
  print(pred)
  return(list(tuned_model,pred))
  
}

classification_compare_accuracy <- function(y_test, linear_kernel_prediction, radial_kernel_prediction, 
                                            polynomial_kernel_prediction, sigmoid_kernel_prediction){
  # ~ 6-10 lines of code
  # Calculate the accuracy for each of the classification methods: 
  # 'svm-linear': linear kernel SVM
  # 'svm-radial': radial kernel SVM
  # 'svm-poly': polynomial kernel SVM
  # 'svm-sigmoid': sigmoid kernel SVM 
  # Return the best method and its accuracy (i.e., method with highest accuracy)
  
  # Inputs:
  # y_test: ground truth dependent variable from test data (factor)
  # linear_kernel_prediction: predictions from linear kernel SVM (factor)
  # radial_kernel_prediction: predictions from radial kernel SVM (factor)
  # polynomial_kernel_prediction: predictions from polynomial kernel SVM (factor)
  # sigmoid_kernel_prediction: predictions from sigmoid kernel SVM (factor)
  
  # Returns:
  # list of three values:
  # First value, of type string, with the name of the best method, should be:
  # 'svm-linear' if linear_kernel_prediction is best
  # 'svm-radial' if radial_kernel_prediction is best
  # 'svm-poly' if polynomial_kernel_prediction is best
  # 'svm-sigmoid' if sigmoid_kernel_prediction is best
  # Second value, of type double, with the corresponding overall accuracy of the best method (on a scale of 100, do not round off)
  # third value, a vector with the overall accuracies of all methods in this order: c(linear-svm's accuracy, radial-svm's accuracy, poly-svm's accuracy, sigmoid-svm's accuracy)
  # Allowed packages: R-base
  # Note that I asked you to implement accuracy calculation - do not use a library for this
  
  # Computing accuracies for all 4 kernal models of SVM
  print(sum(linear_kernel_prediction == y_test))
  print(y_test)
  print(sum(radial_kernel_prediction == y_test))
  print(sum(polynomial_kernel_prediction == y_test))
  print(sum(sigmoid_kernel_prediction == y_test))
  
  linear_acc <- (sum(linear_kernel_prediction == y_test) / length(linear_kernel_prediction)) * 100
  radial_acc <- (sum(radial_kernel_prediction == y_test) / length(radial_kernel_prediction)) * 100
  poly_acc <- (sum(polynomial_kernel_prediction == y_test) / length(polynomial_kernel_prediction)) * 100
  sigm_acc <- (sum(sigmoid_kernel_prediction == y_test) / length(sigmoid_kernel_prediction)) * 100
  
  overall_acc <- c(linear_acc, radial_acc, poly_acc, sigm_acc)
  best_acc <- max(overall_acc)
  
  model_names <- c('svm-linear', 'svm-radial', 'svm-poly', 'svm-sigmoid')
  best_model <- model_names[which.max(overall_acc)]
  
  return(list(best_model,best_acc,overall_acc))
}
