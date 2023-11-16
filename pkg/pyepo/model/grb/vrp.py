#!/usr/bin/env python
# coding: utf-8
"""
Shortest path problem
"""

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from pyepo.model.grb.grbmodel import optGrbModel


class vrpModel(optGrbModel): 
    """
    This class is optimization model for vehicle routing 
    """ 

    def __init__(self, num_customers, demands, capacity, params = {}):
        self.customers = [i for i in range(1, num_customers+1)]
        self.demands = demands # demands of customers  
        # self.demands = {i: demands[i-1] for i in range(1, num_customers+1)}# demands of customers 
        self.capacity = capacity # capacity for a vehicle 
        assert(len(self.demands) == len(self.customers))
        self.nodes = [0] + [i for i in range(1, num_customers+1)]
        self.edges = [(i, j) for i in self.nodes for j in self.nodes if i < j]
        self.rnd_state = np.random.RandomState(1)
        self.params = params 
        super().__init__() 

    @property
    def num_cost(self):
        return len(self.edges)

    def _getModel(self):
        """
        A method to build Gurobi model

        Returns:
            tuple: optimization model and variables
        """
        # ceate a model
        env = gp.Env(params=self.params)
        m = gp.Model("vrp", env=env)
        # varibles
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", vtype=GRB.BINARY) # edge selection 
        u = m.addVars(self.customers, name="v", vtype=GRB.CONTINUOUS) # accumulated demands for a tour 
        # sense
        m.modelSense = GRB.MINIMIZE
        # constraints 
        m.addConstrs(gp.quicksum(x[i, j] for j in self.nodes if j != i) == 1 for i in self.customers)
        m.addConstrs(gp.quicksum(x[i, j] for i in self.nodes if i != j) == 1 for j in self.customers)
        m.addConstrs((x[i, j] == 1) >> (u[i]+self.demands[j-1] == u[j]) for i, j in directed_edges if i != 0 and j != 0)
        m.addConstrs(u[i] >= self.demands[i-1] for i in self.customers)
        m.addConstrs(u[i] <= self.capacity for i in self.customers)

        return m, x

    def setObj(self, c):
        obj = gp.quicksum(c[k] * (self.x[i,j] + self.x[j,i]) for k, (i,j) in enumerate(self.edges))
        self._model.setObjective(obj)

    # todo: use initial solution to warm start the search 
    def solve(self, init_sol): 
        self._model.update() 
        self._model.optimize() 
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i,j) in enumerate(self.edges):
            if self.x[i,j].x > 1e-2: 
                sol[k] += 1
            if self.x[j,i].x > 1e-2:
                sol[k] += 1 
        return sol, self._model.objVal

    def relax(self):
        model_rel = vrpModelRel(len(self.customers), self.demands, self.capacity)
        return model_rel 

class vrpModelRel(vrpModel):
    def _getModel(self):
        # ceate a model
        env = gp.Env(params=self.params)
        m = gp.Model("vrp", env=env)
        # varibles
        directed_edges = self.edges + [(j, i) for (i, j) in self.edges]
        x = m.addVars(directed_edges, name="x", ub=1)
        u = m.addVars(self.customers, name="v", vtype=GRB.CONTINUOUS) # accumulated demands for a tour 
        # sense 
        m.modelSense = GRB.MINIMIZE 
        # constraints 
        m.addConstrs(gp.quicksum(x[i, j] for j in self.nodes if j != i) == 1 for i in self.customers)
        m.addConstrs(gp.quicksum(x[i, j] for i in self.nodes if i != j) == 1 for j in self.customers)
        m.addConstrs((x[i, j] == 1) >> (u[i]+self.demands[j-1] == u[j]) for i, j in directed_edges if i != 0 and j != 0)
        m.addConstrs(u[i] >= self.demands[i-1] for i in self.customers)
        m.addConstrs(u[i] <= self.capacity for i in self.customers)

        return m, x

    def solve(self, init_sol): 
        """
        A method to solve model

        Returns:
            tuple: optimal solution (list) and objective value (float)
        """
        self._model.update()
        self._model.optimize() 
        sol = np.zeros(self.num_cost, dtype=np.uint8)
        for k, (i,j) in enumerate(self.edges):
            if self.x[i,j].x > 1e-2: 
                sol[k] += 1
            if self.x[j,i].x > 1e-2:
                sol[k] += 1 
        return sol, self._model.objVal

    def relax(self):
        raise RuntimeError("Model has already been relaxed.")


