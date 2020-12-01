###########################
# ALDA: hw2.R 
# Instructor: Dr. Thomas Price
# Mention your team details here
# @author: Yang Shi/yangatrue and Krishna Gadiraju/kgadira
#
# Group Number: G32
#
# Team Members:
#   Shreyas Chikkballapur Muralidhara - (schikkb)
#   Mangalnathan Vijayagopal          - (mvijaya2)
#   Nischal Badarinath Kashyap        - (nkashya)
#
###################################################

require(caret)
require(rpart)

# ------- Part A -------

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

knn <- function(x_train, y_train, x_test, distance_method = 'cosine', k = 3){
  # You will be IMPLEMENTING a KNN Classifier here
  
  # You can re-use the calculate_euclidean and calculate_cosine methods from HW1 here.
  # for each row in the distance matrix, calculate the 'k' nearest neighbors
  # and return the most frequently occurring class from these 'k' nearest neighbors.
  
  # INPUT:
  # x_train: Matrix with dimensions: (number_training_samples x number_features)
  # y_train: Vector with length number_training_samples of type factor - refers to the class labels
  # x_test: Matrix with dimensions: (number_test_samples x number_features)
  # k: integer, represents the 'k' to consider in the knn classifier
  # distance_method: String, can be of type ('euclidean' or 'cosine')
  
  # OUTPUT:
  # A vector of predictions of length = number of samples in y_test and of type factor.
  
  # NOTE 1: For cosine, remember, you are calculating similarity, not distance. As a result, K nearest neighbors 
  # k values with highest values from the distance_matrix, not lowest. 
  # For euclidean, you are calculating distance, so you need to consider the k lowest values. 
  
  # NOTE 2:
  # In case of conflicts, choose the class with lower numerical value
  # E.g.: in 4NN, if you have 2 NN of class 1, 2 NN of class 2, there is a conflict b/w class 1 and class 2
  # In this case, you will choose class 1. 
  
  # NOTE 3:
  # You are not allowed to use predefined knn-based packages/functions. Using them will result in automatic zero.
  # Allowed packages: R base, utils
  
  predictions = c() # Vector containing the final predictions of test dataset
  
  # Looping each record of the test dataset
  for (i in c(1:nrow(x_test))){
    
    #Initialize Vector and count variables required to calculate the prediction
    dist = c()
    pos_cnt = 0
    neg_cnt = 0
    
    # looping over the train dataset
    for (j in c(1:nrow(x_train)))
      # Compute the distance of test record with trained model using Euclidean distance
      if (distance_method == 'euclidean')
        # Store the euclidean distance into dist vector
        dist <- c(dist, calculate_euclidean(x_test[i,], x_train[j,]))
    
    # Compute the distance of test record with trained model using Euclidean distance
    else if (distance_method == 'cosine'){
      
      #Perform an inverse to get cosine similarity as calculate_cosine returns cosine distance
      cos_similarity = 1- calculate_cosine(x_test[i,], x_train[j,])
      
      #Store the similarity result in distance vector
      dist <- c(dist, cos_similarity)
    } 
    
    # Store the class variable and the distancevector in dataframe
    dist_df <- data.frame(y_train,dist)
    
    if(distance_method == 'euclidean'){
      #Euclidean - sort the data frame in ascending order
      dist_df <- dist_df[order(dist_df$dist),]
      
      #Retrive the k lowest values and consider them as K nearest neighbours
      dist_df <- dist_df[1:k,1]
    }
    
    else if(distance_method == 'cosine'){
      # Cosine - Sort the data frame in descending order
      dist_df <- dist_df[order(-dist_df$dist),]
      
      #Retrive the k lowest values and consider them as K nearest neighbours
      dist_df <- dist_df[1:k,1]
    }
    
    n <- table(dist_df)
    pred <- names(n)[which(n==max(n))]
    
    predictions <- c(predictions, pred)
  }
  return(predictions)
}

# ------- Part B -------

dtree <- function(x_train, y_train, x_test){
  # You will build a CART decision tree, then use the model to predict class values for a test dataset.
  
  # INPUT:
  # x_train: Matrix with dimensions: (number_training_samples x number_features)
  # y_train: Vector with length number_training_samples of type factor - refers to the class labels
  # x_test: Matrix with dimensions: (number_test_samples x number_features)
  # n_folds: integer, refers to the number of folds for n-fold cross validation
  
  # OUTPUT:
  # A vector of predictions of length = number of samples in y_test and of type factor.
  
  # Allowed packages: rpart, R Base, utils
  
  # HINT1: Make sure to read the documentation for the rpart package. Check out the 'rpart' and 'predict' functions.
  
  # HINT2: I've given you attributes and class labels as separate variables. Do you need to combine them 
  # into a data frame for rpart?
  
  fitted_model <- rpart(y_train ~ ., data= x_train, parms = list(split = "information"))
  
  result <- predict(fitted_model, x_test, type = 'class')
  
  return(result)
  
}

# ------- Part C -------

generate_k_folds <- function(n, k) {
  # This function should randomly assign n datapoints to k folds
  # for cross validation. We will use this function in 
  # k_fold_cross_validation_prediction below.
  
  # INPUT:
  # n: Total number of samples.
  # k: The number of cross validation folds (e.g. 10 for 10-fold CV)
  
  # OUTPUT:
  # A vector representing the "fold" assignment of each row in
  # a dataset with n rows (e.g. 1 = first fold, 2 = second fold)
  # Example output:
  # > generate_k_folds(10, 3)
  # [1] 3 3 1 1 3 2 2 2 1 1
  
  # Note: Your folds should be *random*. You can use the sample() function among others
  # to achieve random assignment
  
  # Tip 1: You may wish to look at the function rep() if you don't want to use loops here.
  # Tip 2: R supports logical indexing. For example, If you input x = 1:5; y=c(1,2,3,2,1); 
  # x[y==2] gives an output of c(2,4).
  
  #val <- sample(rep(1:k, len = nrow(x)))
  return(sample(rep(1:k, len = nrow(x))))
  
}

k_fold_cross_validation_prediction <- function(x, y, k, k_folds, classifier_function) {
  # You will be IMPLEMENTING a cross validation predictor here.
  
  # INPUT:
  # x: Dataframe with dimensions: (number_samples x number_features)
  # y: Vector with length number_samples of type factor - refers to the class labels
  # k: The total fold number of cross validation.
  # k_folds: a vector representing the "fold" assignment of each row in the dataset, generated with
  # the generate_k_folds function
  # classifier_function: The classifier function you wish to use, it can be either knn or dtree. 
  # Note that you don't need to use quote marks for the functions as parameters.
  # OUTPUT:
  # A vector of predicted class values for each instance in x (length = nrow(x)). The ith
  # prediction should correspond to the ith row in x.
  
  predictions = c()
  x$ClassVal <- y
  x$kfolds <- k_folds
  
  
  for(i in c(1:k)){
    
    x_train = x[which(x$kfolds != i),1:(ncol(x)-1)-1]
    
    y_train = x[which(x$kfolds != i),(ncol(x)-1)]
    
    x_test = x[which(x$kfolds == i),1:(ncol(x)-1)-1]
    
    y_test = x[which(x$kfolds == i),(ncol(x)-1)]
    
    pred_class <- classifier_function(x_train, y_train, x_test)
    
    #n <- table(pred_class)
    #pred <- names(n)[which(n==max(n))]
    #print(pred_class)
    
    predictions <- c(predictions, as.character(pred_class))
  }
  return(predictions)
}

# ------- Part D -------

calculate_confusion_matrix <- function(y_pred, y_true){
  # Given the following:
  
  # INPUT:
  # y_pred: predicted class labels (vector, each value of type factor)
  # y_true: ground truth class labels (vector, each value of type factor)
  
  # OUTPUT:
  # a confusion matrix of class "table" with Prediction to the left, and Reference on the top:
  # TN FN 
  # FP TP
  # You can use the caret library to calculate this confusion matrix.
  
  print(table(y_pred,y_true))
  return(table(y_pred,y_true))
}

calculate_accuracy <- function(confusion_matrix){
  # Given the following:
  
  # INPUT:
  # confusion_matrix: A confusion matrix
  
  # OUTPUT:
  # prediction accuracy
  print(sum(diag(confusion_matrix)) / sum(confusion_matrix))
  return(sum(diag(confusion_matrix)) / sum(confusion_matrix))
}

calculate_recall <- function(confusion_matrix){
  # Given the following:
  
  # INPUT:
  # confusion_matrix: A confusion matrix
  
  # OUTPUT:
  # prediction recall
  
  print(confusion_matrix["0","0"]/(confusion_matrix["0","0"]+confusion_matrix["0","1"])) 
  return(confusion_matrix["0","0"]/(confusion_matrix["0","0"]+confusion_matrix["0","1"]))
}

calculate_precision <- function(confusion_matrix){
  # Given the following:
  
  # INPUT:
  # confusion_matrix: A confusion matrix
  
  # OUTPUT:
  # prediction precision
  
  print(confusion_matrix["0","0"]/(confusion_matrix["0","0"]+confusion_matrix["1","0"]))
  return(confusion_matrix["0","0"]/(confusion_matrix["0","0"]+confusion_matrix["1","0"]))
  
}

