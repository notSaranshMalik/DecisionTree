from __future__ import annotations

class Node:

    def __init__(self):
        '''
        Initialises an empty tree structure

        classification is the function for left child nodes
        left is the classification of a leaf node
        left_node is the left node pointer
        right_node is the right node pointer
        '''
        self.classification = None
        self.leaf = None
        self.left_node = None
        self.right_node = None
    
    def getClassification(self) -> callable:
        '''
        Getter class for classification
        '''
        return self.classification

    def setClassification(self, func: callable):
        '''
        Setter class for classification
        '''
        self.classification = func

    def getLeaf(self) -> int:
        '''
        Getter class for leaf
        '''
        return self.leaf

    def setLeaf(self, c: int):
        '''
        Setter class for leaf
        '''
        self.leaf = c

    def getLeftNode(self) -> Node:
        '''
        Getter class for left_node
        '''
        return self.left_node

    def setLeftNode(self, n: Node):
        '''
        Setter class for left_node
        '''
        self.left_node = n

    def getRightNode(self) -> Node:
        '''
        Getter class for right_node
        '''
        return self.right_node

    def setRightNode(self, n: Node):
        '''
        Setter class for right_node
        '''
        self.right_node = n
