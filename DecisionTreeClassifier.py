import numpy as np
from TreeStructure import Node
from EntropyCalculation import *

class decisionTreeClassifier:

    def __init__(self):
        '''
        Basic initialisation of the tree, setting up instance variables
        '''
        self.root = None
        self.test_points = None
        self.test_class = None
        self.step_sizes = None

    def construct_tree(self, test_points: np.ndarray, test_class: np.ndarray,
                       depth: int = 10, step_size: int = 100):
        '''
        Main initialisation method for this decision tree classifier. Takes the 
        input test_points matrix and constructs the tree based on that, up to
        the depth of depth. Step size is the parameter for the intervals in
        the region
        '''

        # Variable setup
        root_stats = np.vstack(( test_points.min(axis=0),
                                 test_points.max(axis=0) ))
        root_test_points = np.full(test_points.shape[0], True)

        # Globals setup
        self.root = Node()
        self.test_points = test_points
        self.test_class = test_class
        self.step_sizes = (root_stats[1] - root_stats[0]) / step_size
        
        # Create the tree
        self.calculateBestSplit(root_test_points, root_stats, 
                                       depth)

    def calculateBestSplit(self, test_points: np.ndarray, stats: np.ndarray,
                           max_d: int, depth: int = 0, cur: Node = None):
        '''
        Calculate the best way to split the test points at the current node
        to maximise information gain, given a set of test points in a range
        given by stats. Max_d is the maximum tree depth, while depth is the 
        current depth of the tree, at 0 by default when it's called for the
        root. Cur is the current node being worked on
        '''

        # Setup for root node
        if cur == None:
            cur = self.root

        # Check for max depth or homogenity, then classify
        values, counts = np.unique(self.test_class[test_points], 
                                   return_counts=True)
        if (len(values) == 1):
            cur.setLeaf(values[0])
            return
        if (depth == max_d):
            ind = np.argmax(counts)
            cur.setLeaf(values[ind])
            return

        # Setup variables to keep track of maximum
        max_gain = 0
        arg_max_gain = None
        max_mask = None

        # Iterate over all features
        for feature in range(stats.shape[1]):

            # Iterate over every finite interval in that feature
            min, max = stats[0, feature], stats[1, feature]
            step_size = self.step_sizes[feature]
            for step in np.arange(min+step_size, max, step_size):

                # Make the divisions
                mask_l = (self.test_points[:, feature] <= step) & test_points
                mask_r = (self.test_points[:, feature] > step) & test_points
                split_l = self.test_class[mask_l]
                split_r = self.test_class[mask_r]

                # Count occurances
                dict_l = dict(zip(*np.unique(split_l, return_counts=True)))
                dict_r = dict(zip(*np.unique(split_r, return_counts=True)))

                # Calculate if this branch increases information gain
                gain = calculateInformationGain(dict_l, dict_r)
                if gain > max_gain:
                    max_gain = gain
                    arg_max_gain = (feature, step)
                    max_mask = (mask_l, mask_r)

        # Modify the current node data
        split_func = lambda x: x[arg_max_gain[0]] <= arg_max_gain[1]
        cur.setClassification(split_func)

        # Create and setup left and right nodes
        node_l = Node()
        cur.setLeftNode(node_l)
        node_r = Node()
        cur.setRightNode(node_r)

        # Recurse onto left and right nodes
        stats_l = np.array(stats, copy=True)
        stats_l[1, arg_max_gain[0]] = arg_max_gain[1]
        self.calculateBestSplit(max_mask[0], stats_l, max_d, depth+1, node_l)
        stats_r = np.array(stats, copy=True)
        stats_r[0, arg_max_gain[0]] = arg_max_gain[1]
        self.calculateBestSplit(max_mask[1], stats_r, max_d, depth+1, node_r)

    def classify(self, data_point: np.ndarray) -> int:
        '''
        Runs the data point through the tree in order to classify it iteratively
        '''

        # Setup the iteration
        cur = self.root

        # Iterate down the tree until you find a leaf
        while (True):

            # If you're at a leaf, return
            if (cur.getLeaf() is not None):
                return cur.getLeaf()
            
            # Else, find which leaf you go down
            classifier = cur.getClassification()
            if classifier(data_point):
                cur = cur.getLeftNode()
            else:
                cur = cur.getRightNode()
                