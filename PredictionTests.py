from sklearn.datasets import *
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTreeClassifier import decisionTreeClassifier

def testDepths(X: np.ndarray, y: np.ndarray):
    # Load up the basic iris data_set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42)

    # Iterate over 6 possible depths
    for depth in range(0, 6):

        # Construct tree
        tree = decisionTreeClassifier()
        tree.construct_tree(X_train, y_train, depth)

        # Predict values for all inputs
        y_hat = tree.classify_many(X_test)

        # Check for accuracy and depth
        print(f"Depth {depth}")
        print(f"Accuracy: {tree.check_accuracy(y_test, y_hat)}")

if __name__ == "__main__":

    # Classification based tests
    (X, y) = load_iris(return_X_y = True)
    testDepths(X, y)

    (X, y) = load_wine(return_X_y = True)
    testDepths(X, y)
    