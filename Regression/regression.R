########################################################
# HW3 
# Instructor: Dr. Thomas Price
# 
# @author: Krishna Gadiraju/kgadira, Yang Shi/yangatrue
#
# Team Members:
#   Shreyas Chikkballapur Muralidhara - (schikkb)
#   Mangalnathan Vijayagopal          - (mvijaya2)
#   Nischal Badarinath Kashyap        - (nkashya)
#
#######################################################

# Write code for regression here
alda_regression <- function(x_train, x_test, y_train, regression_type){
  # Perform regression (linear/lasso)
  
  # Inputs:
  # x_train: training data frame(19 variables, x1-x19)
  # x_test: test data frame(19 variables, x1-x19)
  # y_train: dependent variable, training data (vector, continous type)
  # regression_type: specifies type of regression, string variable, can be of type 'linear' or 'lasso'
  
  # General Information:
  # Instructions for specific regression types:
  # linear: no cross validation
  # lasso: use 10-fold cross validation to determine optimal lambda
  
  # Output:
  # A list with two elements, first element = model generated, second element = predictions on test data (vector) 
  
  # allowed packages: R-base, glmnet
  
  # Function hints: Read the documentation for the functions glmnet, cv.glmnet, predict
  # Ridge and Lasso regression hints: Lambda is the hyperparameter
  if(regression_type == 'linear'){ 
    # ~ 2-3 lines of code
    # write code for building a linear regression model using x_train, y_train
    # Optional: can you use glmnet to do simple linear regression as well?
    # Explore away!  
    # Hint: Think of what the lambda value means for linear regression without regularization.
    
    # Implementing Linear model using lm functionality
    model_lm = lm(formula = y_train ~., as.data.frame(x_train))
    print(summary(model_lm))
    
    # Implementing Linear model using glmnet functionality
    # NOTE: this model does not provide the exact predictions as the lm model, the closest model obtained by tuning the hyper parameter lambda
    model_glmnet = glmnet(x_train,y_train,lambda = 0)
    print(summary(model_glmnet))
    
    # predict using the model
    
    model_pred = c(predict(model_lm,as.data.frame(x_test)))
    
    model_pred_glmnet = c(predict(model_glmnet,x_test))
    
    # RMSE for theglmnet model is 1300.289 which is the closeset implementation of Linear model
    return(model_pred_glmnet)
    
    
  }else{
    # ~ 2-3 lines of code
    # write code for lasso regression here
    # Use 10-fold cross validation *on the training data only* to tune the hyperparameter lambda
    # using MSE (mean squared error) as the measure
    # Hint: use ?cv.glmnet to read more on how lasso uses CV
    
    model_lasso = cv.glmnet(x_train, y_train, type.measure = "mse", nfolds = 10)
    
    print(as.matrix(coef(model_lasso, model_lasso$lambda.1se)))
    # predict on x_test using the model that gives least MSE
    model_pred_lasso = c(predict(model_lasso,x_test))
    
    return(model_pred_lasso)
    
  }
  
}



