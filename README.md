# DecisionTree

## Overview
A basic decision tree classification algorithm made from scratch in Python, using Numpy to process the data

## Creating a tree

### Basic format
```
tree = decisionTreeClassifier()
tree.construct_tree(X_train, y_train)
```

### Optional construct_tree parameters
1) `depth` for the maximum tree depth (default 10)
2) `step_size` for the number of finite intervals per feature (default 100)

##  Predicting classification for a data point
```
prediction = tree.classify(test_point)
```

## Accuracy
1) 96% accuracy on iris dataset with depth 3
2) 91% accuracy on wine dataset with depth 3
