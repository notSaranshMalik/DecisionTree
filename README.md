# DecisionTree

## Overview
A basic decision tree classification algorithm made from scratch in Python, using Numpy to process the data

## Creating a tree

### Basic format
```
tree = decisionTreeClassifier()
tree.construct_tree(X_train, y_train)
```
- X_train is a Numpy matrix of size (n, w), where there are n data points and w features per data point

- y_train is a Numpy array of size n

### Optional construct_tree parameters
1) `depth` for the maximum tree depth (default 10)
2) `step_size` for the number of finite intervals per feature (default 100)
3) `regress` for if the construction should be for regression instead of classification (default False)

##  Predicting classification for a data point
```
prediction = tree.classify(test_point)
```
- test_point is a Numpy array of size w
- prediction is an integral value

##  Predicting classification for a data set
```
predictions = tree.classify_many(X_test)
```
- X_test is a Numpy matrix of size (m, w)
- predictions is a Numpy array of size m

##  Checking accuracy
```
accuracy_rate = tree.check_accuracy(y_test, predictions)
```
- y_test is a Numpy array of size m

## Accuracy
1) 96% accuracy on iris dataset with depth 3
2) 91% accuracy on wine dataset with depth 3
