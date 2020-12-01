## Problem Description

You are given the following files:
* pima-indians-diabetes.csv: CSV file with 100 rows, 9 columns. The first 8 columns refer to features, final column (called `Class') refers to the Class variable. 
You have 2 classes: f0, 1g.  
* Data is from Johannes et al. [1], the 8 features are number of times pregnant, glucose concentration, blood pressure, skinfold thickness, serum insulin, body mass 
index, diabetes pedigree function, age. The label indicates whether the patient was diagnosed with diabetes (1) or not (0).  

**Part A: KNN Classification: (Implement)** You are tasked with implementing the KNN algorithm using the dataset provided to you. You will be implementing the KNN classifier 
in hw2.R using the knn function. The input and output variables, and their formats are explained in detail knn function documentation { read this carefully before starting.    

**Part B: Decision Tree (Library):** Implement a decision tree classifier using the rpart library. Use information gain (IG) to choose the best attribute. You should train 
a model using the provided training data and then predict the classes of the provided test data. Code for this question has to be written in the function dtree. Please note 
that by default, rpart will automatically tune its model parameters.

**Part C: Cross Validation (Implementation):** Write a function for k-fold cross-validation. It should split the whole dataset into k folds, then predict the class values 
for each fold by training on the other k􀀀1 folds. It should return a vector of predicted class values, one for each row in original the dataset. You will do this in 2 steps. 
First you will write a generate k folds function that randomly divides the dataset into k folds. Then you will write a k fold cross validation prediction function that uses 
these folds to make predictions. For each fold i, the classifier should be trained on all but the ith folds and used to predict class values for the i-th fold. Explanations 
of each function are in their respective documentation.

**Part D: Evaluation (Library + implementation):** Write functions calculate confusion matrix, which that takes in the predicted class labels of type vector (factor) and 
ground truth labels of type vector (factor), and outputs a confusion matrix. Then write calculate accuracy, calculate recall, calculate precision, which take in that matrix 
and output accuracy, precision and recall, respectively. The input, output formats, as well as allowed packages are described as comments under the function.

## References  
[1] R. S. Johannes, \Using the g algorithm to forecast the onset of diabetes mellitus," Johns Hopkins APL Technical Digest, vol. 10, pp. 262{266, 1988.