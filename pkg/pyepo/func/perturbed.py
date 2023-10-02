#!/usr/bin/env python
# coding: utf-8
"""
Perturbed optimization function
"""

import numpy as np
import torch
from torch.autograd import Function

from pyepo import EPO
from pyepo.func.abcmodule import optModule
from pyepo.utlis import getArgs
from pyepo.func.utlis import _solve_in_pass


class perturbedOpt(optModule):
    """
    An autograd module for differentiable perturbed optimizer, in which random
    perturbed costs are sampled to optimize.

    For the perturbed optimizer, the cost vector need to be predicted from
    contextual data and are perturbed with Gaussian noise.

    The perturbed optimizer differentiable in its inputs with non-zero Jacobian.
    Thus, allows us to design an algorithm based on stochastic gradient descent.

    Reference: <https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, processes=1,
                 seed=135, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            seed (int): random state seed
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
        # random state
        self.rnd = np.random.RandomState(seed)
        # build optimizer
        self.ptb = perturbedOptFunc()

    def forward(self, pred_cost, sol_index):
        """
        Forward pass
        """
        sols = self.ptb.apply(pred_cost, sol_index, self.optmodel, self.n_samples,
                              self.sigma, self.processes, self.pool, self.rnd,
                              self.solve_ratio, self)
        return sols


class perturbedOptFunc(Function):
    """
    A autograd function for perturbed optimizer
    """

    @staticmethod
    def forward(ctx, pred_cost, sol_index, optmodel, n_samples, sigma,
                processes, pool, rnd, solve_ratio, module):
        """
        Forward pass for perturbed

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            rnd (RondomState): numpy random state
            solve_ratio (float): the ratio of new solutions computed during training
            module (optModule): perturbedOpt module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        idx = sol_index.detach().to("cpu").numpy() 
        # sample perturbations
        noises = rnd.normal(0, 1, size=(n_samples, *cp.shape))
        ptb_c = cp + sigma * noises
        ptb_sols = [[None] * len(cp)] * n_samples
        # solve with perturbation
        if module.solpool != None: 
            for i in range(n_samples): 
                ptb_sols[i], _ = module.solpool.cache_in_pass(ptb_c[i], idx, optmodel.modelSense)
        
        if np.random.uniform() <= solve_ratio:
            for i in range(n_samples):
                ptb_sols[i], _ = _solve_in_pass(ptb_c[i], optmodel, processes, pool, ptb_sols[i])
                if module.solpool != None: 
                     # todo: handle last solution caching with perturbations
                    module.solpool.update_cache(idx, ptb_sols[i])

        ptb_sols = np.stack(ptb_sols, axis=0).transpose(1,0,2)
        
        # solution expectation
        e_sol = ptb_sols.mean(axis=1)
        # convert to tensor
        noises = torch.FloatTensor(noises).to(device)
        ptb_sols = torch.FloatTensor(ptb_sols).to(device)
        e_sol = torch.FloatTensor(e_sol).to(device)
        # save solutions
        ctx.save_for_backward(ptb_sols, noises)
        # add other objects to ctx
        ctx.optmodel = optmodel
        ctx.n_samples = n_samples
        ctx.sigma = sigma
        return e_sol

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for perturbed
        """
        ptb_sols, noises = ctx.saved_tensors
        optmodel = ctx.optmodel
        n_samples = ctx.n_samples
        sigma = ctx.sigma
        grad = torch.einsum("nbd,bn->bd",
                            noises,
                            torch.einsum("bnd,bd->bn", ptb_sols, grad_output))
        grad /= n_samples * sigma
        return grad, None, None, None, None, None, None, None, None, None


class perturbedFenchelYoung(optModule):
    """
    An autograd module for Fenchel-Young loss using perturbation techniques. The
    use of the loss improves the algorithmic by the specific expression of the
    gradients of the loss.

    For the perturbed optimizer, the cost vector need to be predicted from
    contextual data and are perturbed with Gaussian noise.

    The Fenchel-Young loss allows to directly optimize a loss between the features
    and solutions with less computation. Thus, allows us to design an algorithm
    based on stochastic gradient descent.

    Reference: <https://papers.nips.cc/paper/2020/hash/6bb56208f672af0dd65451f869fedfd9-Abstract.html>
    """

    def __init__(self, optmodel, n_samples=10, sigma=1.0, processes=1,
                 seed=135, solve_ratio=1, dataset=None):
        """
        Args:
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            seed (int): random state seed
            solve_ratio (float): the ratio of new solutions computed during training
            dataset (None/optDataset): the training data
        """
        super().__init__(optmodel, processes, solve_ratio, dataset)
        # number of samples
        self.n_samples = n_samples
        # perturbation amplitude
        self.sigma = sigma
        # random state
        self.rnd = np.random.RandomState(seed)
        # build optimizer
        self.pfy = perturbedFenchelYoungFunc()

    def forward(self, pred_cost, true_sol, sol_index, reduction="mean"):
        """
        Forward pass
        """
        loss = self.pfy.apply(pred_cost, true_sol, sol_index, self.optmodel, self.n_samples,
                              self.sigma, self.processes, self.pool, self.rnd,
                              self.solve_ratio, self)
        # reduction
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
        elif reduction == "none":
            loss = loss
        else:
            raise ValueError("No reduction '{}'.".format(reduction))
        return loss


class perturbedFenchelYoungFunc(Function):
    """
    A autograd function for Fenchel-Young loss using perturbation techniques.
    """

    @staticmethod
    def forward(ctx, pred_cost, true_sol, sol_index, optmodel, n_samples, sigma,
                processes, pool, rnd, solve_ratio, module):
        """
        Forward pass for perturbed Fenchel-Young loss

        Args:
            pred_cost (torch.tensor): a batch of predicted values of the cost
            true_sol (torch.tensor): a batch of true optimal solutions
            optmodel (optModel): an PyEPO optimization model
            n_samples (int): number of Monte-Carlo samples
            sigma (float): the amplitude of the perturbation
            processes (int): number of processors, 1 for single-core, 0 for all of cores
            pool (ProcessPool): process pool object
            rnd (RondomState): numpy random state
            solve_ratio (float): the ratio of new solutions computed during training
            module (optModule): perturbedFenchelYoung module

        Returns:
            torch.tensor: solution expectations with perturbation
        """
        # get device
        device = pred_cost.device
        # convert tenstor
        cp = pred_cost.detach().to("cpu").numpy()
        w = true_sol.detach().to("cpu").numpy()
        idx = sol_index.detach().to("cpu").numpy() 
        # sample perturbations
        noises = rnd.normal(0, 1, size=(n_samples, *cp.shape))
        ptb_c = cp + sigma * noises
        ptb_sols = [[None] * len(cp)] * n_samples
        # solve with perturbation
        if module.solpool != None: 
            for i in range(n_samples):
                ptb_sols[i], _ = module.solpool.cache_in_pass(ptb_c[i], idx, optmodel.modelSense)
        
        if np.random.uniform() <= solve_ratio:
            for i in range(n_samples):
                ptb_sols[i], _ = _solve_in_pass(ptb_c[i], optmodel, processes, pool, ptb_sols[i])
                if module.solpool != None: 
                     # todo: handle last solution caching with perturbations
                    module.solpool.update_cache(idx, ptb_sols[i])
        ptb_sols = np.stack(ptb_sols, axis=0).transpose(1,0,2)

        # solution expectation
        e_sol = ptb_sols.mean(axis=1)
        # difference
        if optmodel.modelSense == EPO.MINIMIZE:
            diff = w - e_sol
        if optmodel.modelSense == EPO.MAXIMIZE:
            diff = e_sol - w
        # loss
        loss = np.sum(diff**2, axis=1)
        # convert to tensor
        diff = torch.FloatTensor(diff).to(device)
        loss = torch.FloatTensor(loss).to(device)
        # save solutions
        ctx.save_for_backward(diff)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for perturbed Fenchel-Young loss
        """
        grad, = ctx.saved_tensors
        grad_output = torch.unsqueeze(grad_output, dim=-1)
        return grad * grad_output, None, None, None, None, None, None, None, None, None, None
