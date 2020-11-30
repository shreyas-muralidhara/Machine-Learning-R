## Problem Description

**Dataset You are given the following dataset(s):**  
Iris dataset [1]. You are provided a subset of the Iris dataset. Each line in iris:csv represents a five
element vector, representing a single sample from your dataset. Each value if the first four columns
are the attributes of the sample, respectively "sepal length", "sepal width", "petal length", "petal
width". The fifth column is the class values, specifying which class of iris plant the sample is from.
In total, there are 60 sample points.

**Part 1: Distance Measurement** Before doing analysis, you will need to look through the data file,
and write a function named read data to read the dataset in as a dataframe.  
      1a) Using the data provided in iris:csv, you are to implement the distance/similarity measurements
        defined below. The inputs will be two vectors of the same length:

      (a) euclidean: euclidean(P, Q) = pP i(Pi 􀀀 Qi)2, where P and Q are vectors of equal length.  
      (b) cosine: cosine(P, Q) = 1 - Pi Pi*Qi /||P||*||Q|| , where P and Q are vectors of equal length, and ||P|| = pP i P2 i.  
      (c) L1: L1(P, Q) = max i jPi 􀀀 Qij, where P and Q are vectors of equal length.  

1b) Your goal is to investigate how useful each distance function is in telling apart owers of different species. Ideally, a distance measure should be large for owers 
of different species, and relatively smaller for ower of the same species.To help you with this task, we have provided you with a function: inter intra species dist. This
function calculates the distance between each ower in the provided iris dataset, using the specified distance function. It then averages the following properties for each 
ower species: 
```
  (a) mean intra dis: The average distance between owers of this (same) species.  
  (b) mean inter dis: The average distance between owers of this species and other (different) species.  
  (c) ratio: The ratio of mean intra dis / mean inter dis  
```


**Part 2: Principal Component Analysis** In this part, you will need to implement a function to calculate the principal components (PCs) of a dataset in the function principal component analysis.
You are encouraged to leverage the existing function in R, which is prcomp. The input of this function would be an iris dataframe, and you may need to note that the final column is a nominal value, which
cannot included in the calculation of PCA. The output of the function is a vector of the weights (the eigenvector) of first principal component.

**Part 3: Principal Component Distance** We want to see whether the first PC meaningfully captures the differences between iris species. Implement the pc1 distance function, which takes in two data objects (vectors) and a set of PC weights,
and returns the distance between those two vectors in the dimension of the first PC, i.e. the absolute difference between their PC values.  

**Part 4: Comparing Distances** Now we want to compare our PC1 distance to traditional euclidean distance. In your PDF, use the inter intra species dist function to answer the following question: Which of the two distance
metrics (euclidean, PC1) is most useful for differentiating iris species? Why do you think it is most useful?

**Allowed Packages:** R Base, plyr.

## References
[1] R. A. Fisher, \The use of multiple measurements in taxonomic problems," Annals of eugenics, vol. 7, no. 2, pp. 179{188, 1936.