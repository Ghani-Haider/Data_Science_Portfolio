# Text Classification using Naive Bayes

Given a dataset of textual summary of medical queries classified into five different categories. I 
have built my own naïve Bayes classifier to predict these categories for future queries. The task 
includes:
- Implemented Learn() method that takes training and test data (in the form of word 
vectors) and learns a Naïve Bayes classifier (i.e. all probabilities that a Naïve Bayes classifier 
needs).
- Modified classifier such that it applies Laplacian smoothing while learning conditional probabilities to 
avoid zero values. 
- Implemented Predict() that takes a learned classifier and test data and returns 
predicted values.
- Implemented Evaluate() that takes actual and predicted labels and returns precision, 
recall, f-measure. The method will also display confusion matrix.
