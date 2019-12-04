"""This module define metrics for measuring the distance between 2 sets of sample.
The sets do not need to be the same size.
"""

import numpy as np
from sklearn.metrics import pairwise
from scipy import spatial
from scipy import optimize


def mean_l2(l1, l2):
    """ Return the L2 distance between the mean point of l1 and the mean point of l2.

    :param l1: The first array of samples.
    :param l2: The second array of samples.
    :return: The L2 distance between the mean point of each set.
    """
    return np.linalg.norm(np.mean(l1, 0) - np.mean(l2, 0))


def max_l2(l1, l2):
    """ Return the L2 distance between the max point of l1 and the max point of l2.
    Max points are computed by taking the maximum value per feature across all sample in a set.

    :param l1: The first array of samples.
    :param l2: The second array of samples.
    :return: The L2 distance between the max point of each set.
    """
    return np.linalg.norm(np.max(l1, 0) - np.max(l2, 0))


def mean_cosine(l1, l2):
    """ Return the cosine distance between the mean point of l1 and the mean point of l2.
    Cosine distance is 1 - cosine similarity, where the cosine similarity between a sample x and a sample y
    is given by <x,y> / ||x||.||y|| (euclidian dot product and norms).

    :param l1: The first array of samples.
    :param l2: The second array of samples.
    :return: The cosine distance between the mean point of each set.
    """
    m1 = np.mean(l1, 0)
    m2 = np.mean(l2, 0)
    return 1 - np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2))


def max_cosine(l1, l2):
    """ Return the cosine distance between the max point of l1 and the max point of l2.
    Max points are computed by taking the maximum value per feature across all sample in a set.
    Cosine distance is 1 - cosine similarity, where the cosine similarity between a sample x and a sample y
    is given by <x,y> / ||x||.||y|| (euclidian dot product and norms).

    :param l1: The first array of samples.
    :param l2: The second array of samples.
    :return: The L2 distance between the max point of each set.
    """
    m1 = np.max(l1, 0)
    m2 = np.max(l2, 0)
    return 1 - np.dot(m1, m2) / (np.linalg.norm(m1) * np.linalg.norm(m2))


def minimal_assignment_l2(l1, l2):
    """ Return the minimal assignment cost between points of l1 and l2 where the costs are l2 distance.
    Minimal assignment between 2 sets of elements X and Y, given a cost function C(x,y), consists in finding,
    a set S of pairs of element from X and Y such that:
    - for every (x,y) in S, x is in X and y is in Y (all pairs are made of one element of x and one element of Y)
    - if (x,y1) in S and (x, y2) in S then y1 = y2 (a element of X cannot be paired with 2 distinct elements of Y)
    - if (x1,y) in S and (x2, y) in S then x1 = x2 (a element of Y cannot be paired with 2 distinct elements of X)
    - |S| = min(|X|, |Y|) (Every elements of at least one of the sets are paired)

    :param l1: The first array of samples.
    :param l2: The second array of samples.
    :return: The cost of the minimal assignement, equal to the sum of the l2 distances between of pair of points.
    """
    distance_matrix = spatial.distance_matrix(l1, l2)
    x_list, y_list = optimize.linear_sum_assignment(distance_matrix)
    return distance_matrix[x_list, y_list].sum()


def minimal_assignment_cosine(l1, l2):
    """ Return the minimal assignment cost between points of l1 and l2 where the costs are l2 distance.
    Minimal assignment between 2 sets of elements X and Y, given a cost function C(x,y), consists in finding,
    a set S of pairs of element from X and Y such that:
    - for every (x,y) in S, x is in X and y is in Y (all pairs are made of one element of x and one element of Y)
    - if (x,y1) in S and (x, y2) in S then y1 = y2 (a element of X cannot be paired with 2 distinct elements of Y)
    - if (x1,y) in S and (x2, y) in S then x1 = x2 (a element of Y cannot be paired with 2 distinct elements of X)
    - |S| = min(|X|, |Y|) (Every elements of at least one of the sets are paired)

    Cosine distance is 1 - cosine similarity, where the cosine similarity between a sample x and a sample y
    is given by <x,y> / ||x||.||y|| (euclidian dot product and norms).

    :param l1: The first array of samples.
    :param l2: The second array of samples.
    :return: The cost of the minimal assignement, equal to the sum of the l2 distances between of pair of points.
    """
    distance_matrix = pairwise.cosine_distances(l1, l2)
    x_list, y_list = optimize.linear_sum_assignment(distance_matrix)
    return distance_matrix[x_list, y_list].sum()


if __name__ == "__main__":
    l1 = [np.array([1, 2, 3]), np.array([3, 1, 1])]
    l2 = [np.array([3, 1, 1]), np.array([-1, 5, -1]), np.array([1.1, 2.2, 3.3])]

    print("L2 mean: {}".format(mean_l2(l1, l2)))
    print("L2 max: {}".format(max_l2(l1, l2)))
    print("Cosine mean: {}".format(mean_cosine(l1, l2)))
    print("Cosine max: {}".format(max_cosine(l1, l2)))
    print("Minimal L2 assignment: {}".format(minimal_assignment_l2(l1, l2)))
    print("Minimal cosine assignment: {}".format(minimal_assignment_cosine(l1, l2)))
