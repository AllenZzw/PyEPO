#!/usr/bin/env python
# coding: utf-8
"""
Abstract autograd optimization module
"""

from abc import abstractmethod
import multiprocessing as mp
from pathos.multiprocessing import ProcessingPool

import numpy as np
from torch import nn

from pyepo.data.dataset import optDataset
from pyepo.model.opt import optModel
from pyepo import EPO

class SolCahce():
    def __init__(self, dataset, mode="best"):
        if not isinstance(dataset, optDataset): # type checking
            raise TypeError("dataset is not an optDataset")
        self.cache = None
        self.mode = mode
        if self.mode == "best":
            self.cache = np.unique(dataset.sols.copy(), axis=0) # remove duplicate
        elif self.mode == "last":
            self.cache = dataset.sols.copy() 
        else: 
            raise("Unregconized mode in SolCache")
    
    def cache_in_pass(self, cp, index, modelSense):
        obj, sol = None, None 
        if self.mode == "best": 
            ins_num = len(cp) # number of instance
            solpool_obj = cp @ self.cache.T # best solution in pool
            if modelSense == EPO.MINIMIZE:
                ind = np.argmin(solpool_obj, axis=1)
            if modelSense == EPO.MAXIMIZE:
                ind = np.argmax(solpool_obj, axis=1)
            obj = np.take_along_axis(solpool_obj, ind.reshape(-1,1), axis=1).reshape(-1)
            sol = self.cache[ind]
        elif self.mode == "last": 
            sol = self.cache[index]
            obj = np.sum(cp * sol, axis=1)
        return sol, obj

    def update_cache(self, index, sol):
        if self.mode == "best": 
            self.cache = np.concatenate((self.cache, sol))
            self.cache = np.unique(self.cache, axis=0)
        elif self.mode == "last": 
            for i in range(len(index)): 
                self.cache[index[i]] = sol[i]

class optModule(nn.Module):
    """
        An abstract module for the learning to rank losses, which measure the difference in how the predicted cost
        vector and the true cost vector rank a pool of feasible solutions.
    """
    def __init__(self, optmodel, processes=1, solve_ratio=1, dataset=None, mode="best"):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__()
        # optimization model
        if not isinstance(optmodel, optModel):
            raise TypeError("arg model is not an optModel")
        self.optmodel = optmodel
        # number of processes
        if processes not in range(mp.cpu_count()+1):
            raise ValueError("Invalid processors number {}, only {} cores.".
                format(processes, mp.cpu_count()))
        self.processes = mp.cpu_count() if not processes else processes
        # single-core
        if processes == 1:
            self.pool = None
        # multi-core
        else:
            self.pool = ProcessingPool(processes)
        print("Num of cores: {}".format(self.processes))
        # solution pool
        self.solve_ratio = solve_ratio
        if (self.solve_ratio < 0) or (self.solve_ratio > 1):
            raise ValueError("Invalid solving ratio {}. It should be between 0 and 1.".
                format(self.solve_ratio))
        self.solpool = None
        if dataset != None:
            if not isinstance(dataset, optDataset): # type checking
                raise TypeError("dataset is not an optDataset")
            self.solpool = SolCahce(dataset, mode)

    @abstractmethod
    def forward(self, pred_cost, true_cost, reduction="mean"):
        """
        Forward pass
        """
        # convert tensor
        pass
