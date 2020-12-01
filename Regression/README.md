## Problem Description

i. **Dataset description:** You are provided a dataset with 20 variables. Variables x1 - x19
refer to the independent variables, while variable y is your dependent variable. Training
data is stored in the file data/regression-train.csv, and test data is stored in the file
data/regression-test.csv.  
ii. In this exercise, you will apply linear regression and Lasso regression methods to the dataset
supplied to you, and then compare their results to determine whether Lasso regression is
needed for this dataset:  
iii. **Learning:** You will write code in the function alda regression() to train simple
linear regression and lasso regression models. Detailed instructions for implementation and
allowed packages have been provided in hw3.R. Note that for the lasso regression model, you
will be using crossvalidation to tune the lambda hyperparameter.  