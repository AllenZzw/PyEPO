#!/usr/bin/env python
# coding: utf-8
"""
Synthetic data for maximum cut problem 
"""

import numpy as np
import networkx as nx 


def genData(num_data, num_features, num_nodes, prob=0.1, deg=1, noise_width=0, seed=135):
    """
    A function to generate synthetic data and features for maximum independent set problem 

    Args:
        num_data (int): number of data points
        num_features (int): dimension of features
        num_nodes (int): number of nodes
        prob (float): probability of an edge between nodes 
        deg (int): data polynomial degree
        noise_width (float): half witdth of data random noise
        seed (int): random seed

    Returns:
        tuple: data features (np.ndarray), costs (np.ndarray)
    """
    # positive integer parameter
    if type(deg) is not int:
        raise ValueError("deg = {} should be int.".format(deg))
    if deg <= 0:
        raise ValueError("deg = {} should be positive.".format(deg))
    # set seed
    rnd = np.random.RandomState(seed)
    # number of data points
    n = num_data
    # dimension of features
    p = num_features
    # number of nodes
    m = num_nodes
    # random regular graph
    graph = nx.erdos_renyi_graph(m, prob, seed)
    # random matrix parameter B
    B = rnd.binomial(1, 0.5, (m, p))
    # feature vectors
    x = rnd.normal(0, 1, (n, p))
    # cost vectors 
    c = np.zeros((n, m))
    for i in range(n):
        # cost without noise
        ci = (np.dot(B, x[i].reshape(p, 1)).T / np.sqrt(p) + 3) ** deg + 1
        # rescale
        ci /= 3.5 ** deg
        # noise
        epislon = rnd.uniform(1 - noise_width, 1 + noise_width, m)
        ci *= epislon
        c[i, :] = ci

    # rounding
    c = np.around(c, decimals=4)
    return x, c, graph
