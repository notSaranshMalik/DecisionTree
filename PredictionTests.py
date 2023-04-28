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
        y_hat = np.zeros(y_test.size)
        for point in range(y_test.size):
            y_hat[point] = tree.classify(X_test[point])

        # Check for accuracy and depth
        print(f"Depth {depth}")
        print(f"Accuracy: {round((y_test == y_hat).sum() / y_test.size, 2)}")

if __name__ == "__main__":

    # Classification based tests
    (X, y) = load_iris(return_X_y = True)
    testDepths(X, y)

    (X, y) = load_wine(return_X_y = True)
    testDepths(X, y)
    