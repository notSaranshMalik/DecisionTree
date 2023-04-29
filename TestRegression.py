from sklearn.datasets import *
from sklearn.model_selection import train_test_split
import numpy as np
from DecisionTreeClassifier import decisionTreeClassifier

def testRegress(X: np.ndarray, y: np.ndarray):

    # Load up the basic iris data_set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    # Construct tree
    tree = decisionTreeClassifier()
    tree.construct_tree(X_train, y_train, 3, regress=True)

    # Predict values for all inputs
    y_hat = tree.classify_many(X_test)

    # Check for accuracy and depth
    print(np.vstack((y_test, y_hat)).T)
    print(f"RMSE: {tree.check_closeness(y_test, y_hat)}")

if __name__ == "__main__":

    # Regression based tests
    (X, y) = load_diabetes(return_X_y = True) 
    testRegress(X, y) # Tends to overfit past depth 3
    