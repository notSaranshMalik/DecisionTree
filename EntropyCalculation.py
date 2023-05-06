import math
import numpy as np

def calculateEntropy(d: dict, rounded = True) -> float:
    '''
    Calculate the Shannon Entropy of a given dictionary d
    Rounds to 3dp if rounded
    '''

    # Calculate the total number of elements in the dictionary
    total_elements = 0
    for v in d.values():
        total_elements += v

    # Sum up individual entropy values
    total_entropy = 0
    for v in d.values():
        prob = v / total_elements
        total_entropy += (prob * math.log2(prob))

    # Take the negative to achieve entropy, and return rounded
    total_entropy = -total_entropy
    if rounded:
        total_entropy = round(total_entropy, 3)
    return total_entropy

def calculateInformationGain(d1: dict, d2: dict) -> float:
    '''
    Calculate the Information gain given a specific boundary split.
    '''

    # Calculate the individual entropies for the sections
    entropy1 = calculateEntropy(d1, rounded = False)
    entropy2 = calculateEntropy(d2, rounded = False)

    # Calculate the total entropy
    d1_size = 0
    d2_size = 0
    for v in d1.values():
        d1_size += v
    for v in d2.values():
        d2_size += v
    total_size = d1_size + d2_size
    d1_fraction = d1_size/total_size
    d2_fraction = d2_size/total_size
    total_entropy = d1_fraction * entropy1 + d2_fraction * entropy2

    # Calculate the original dictionary set with all elements
    full_dict = {}
    for x in set(d1).union(d2):
        full_dict[x] = d1.get(x, 0) + d2.get(x, 0)

    # Calculate and return the information gain
    original_entropy = calculateEntropy(full_dict, rounded = True)
    return round(original_entropy - total_entropy, 3)

def cartRegress(a1: np.ndarray, a2: np.ndarray):
    '''
    Calculate information gain through total squared error (SE) based on 
    difference between original SE and sum of new SE
    '''

    # Check for items
    if (a1.size == 0 or a2.size == 0):
        return 0

    # Calculate full set
    a_total = np.append(a1, a2)

    # Calculate and return gain
    SE_total = a_total - np.median(a_total)
    SE_1 = a1 - np.median(a1)
    SE_2 = a2 - np.median(a2)
    return SE_total @ SE_total - SE_1 @ SE_1 - SE_2 @ SE_2
