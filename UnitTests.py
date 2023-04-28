import unittest
import numpy as np
from EntropyCalculation import *
from TreeStructure import *

class EntropyCalculation(unittest.TestCase):

    def testCalculateEntropy(self):

        d1 = {"a": 3, "b": 3}
        self.assertEqual(calculateEntropy(d1), 1.000)

        d2 = {"a": 2, "b": 4, "c": 5}
        self.assertEqual(calculateEntropy(d2), 1.495)

    def testCalculateInformationGain(self):

        d1a = {"a": 4, "b": 1}
        d1b = {"a": 1, "b": 3}
        self.assertEqual(calculateInformationGain(d1a, d1b), 0.229)

        d2a = {"a": 3, "b": 1}
        d2b = {"a": 2, "b": 3}
        self.assertEqual(calculateInformationGain(d2a, d2b), 0.091)

class TreeStructure(unittest.TestCase):

    def testSetClassification(self):

        root = Node()

        is_one = lambda x: x == 1
        root.setClassification(is_one)
        self.assertTrue(root.classification == is_one)

    def testGetClassification(self):

        root = Node()

        is_one = lambda x: x == 1
        root.classification = is_one
        self.assertTrue(root.getClassification() == is_one)

    def testSetLeaf(self):

        root = Node()

        root.setLeaf(2)
        self.assertTrue(root.leaf == 2)

    def testGetLeaf(self):

        root = Node()

        root.leaf = 2
        self.assertTrue(root.getLeaf() == 2)

    def testSetLeftNode(self):

        root = Node()

        n2 = Node()
        root.setLeftNode(n2)
        self.assertTrue(root.left_node == n2)

    def testGetLeftNode(self):

        root = Node()

        n2 = Node()
        root.left_node = n2
        self.assertTrue(root.getLeftNode() == n2)

    def testSetRightNode(self):

        root = Node()

        n2 = Node()
        root.setRightNode(n2)
        self.assertTrue(root.right_node == n2)

    def testGetRightNode(self):

        root = Node()

        n2 = Node()
        root.right_node = n2
        self.assertTrue(root.getRightNode() == n2)

if __name__ == '__main__':
    unittest.main()